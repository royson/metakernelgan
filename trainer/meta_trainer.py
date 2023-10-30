import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.trainer import Trainer
import trainer.utils as tutils
from trainer import kernelgan
import data.utils as dutils
from utils import cycle
import numpy as np

import logging
logger = logging.getLogger(__name__)

class MetaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ### Meta-algorithms for each model
        self.meta_models = {}
        for model_typ, model in self.models.items():
            opt_args = model.optim_args
            self.meta_models[model_typ] = tutils.get_meta_algorithm(args=self.args,
                        algo=opt_args.meta_algorithm.upper(), 
                        model=model, 
                        task_lr=opt_args.task_lr, 
                        first_order=opt_args.first_order,
                        task_opt=opt_args.task_opt,
                        task_opt_args=opt_args.task_opt_args,
                        gradient_clip=opt_args.task_gradient_clip,
                        device=self.device)

        ### Training optimizers and schedulers
        if not self.args.train.test_only:            
            self.degradation_operation = self.args.degradation_operations.train
            self.steps = self.args.optim.meta_steps

            # Outer loop optimizers
            for meta_model_typ, meta_model in self.meta_models.items():
                opt_args = meta_model.optim_args
                self.optimizers[meta_model_typ] = tutils.get_optimizer(meta_model, opt_args.meta_opt, \
                                                opt_args.meta_lr, opt_args.meta_opt_args)
                
                # Outer loop schedulers
                if opt_args.meta_lr_decay is not None:
                    self.schedulers[meta_model_typ] = torch.optim.lr_scheduler.MultiStepLR(self.optimizers[meta_model_typ], 
                            milestones=opt_args.meta_lr_decay, gamma=0.1)
                    logger.info(f'[*] {meta_model.model_name} - Multi-Step LR Scheduler set for {opt_args.meta_lr_decay}')

        ### Model Loading
        self.load_previous_ckp(models=self.meta_models)
        
        ### Saving initial model
        if self.existing_step == 0 and not self.args.train.test_only:
            logger.debug('Save initial meta-training model')
            self.save_ckp(self.existing_step, models=self.meta_models) 
 
        ### GAN
        if self.gan:
            self.lsgan_loss = kernelgan.GANLoss(self.dis_output_ps, batch_size=self.args.optim.task_batch_size, device=self.device)
            self.outer_lsgan_loss = self.args.optim.outer_lsgan_loss
            logger.info(f'[*] LSGAN Loss is enabled. Dis input size: {self.dis_input_ps}. Dis output size: {self.dis_output_ps}. \
                            Outer-loop LSGAN Loss: {self.outer_lsgan_loss}.') 
        else:
            self.lsgan_loss = False
            self.outer_lsgan_loss = False
          
    def sample_task(self):
        raise NotImplementedError()

    def train(self):
        self.train_iter = iter(cycle(self.train_dataloader))
        assert self.existing_step < self.steps, 'Training is done.'
        logger.info(f'{self.steps - self.existing_step} steps left.')

        for i in range(self.existing_step, self.steps):
            for model_typ, model in self.meta_models.items():
                model.train()
                self.optimizers[model_typ].zero_grad()

            loss = self.meta_train(i)

            for opt in self.optimizers.values():
                opt.step()
            for sch in self.schedulers.values():
                sch.step()

            log_msg = f'[Step {i+1}] Train error: {loss}, '            
            logger.debug(log_msg)

            if (i + 1) % self.args.optim.save_every == 0:
                logger.info(log_msg)
                self.save_ckp(i + 1, models=self.meta_models)

            self.log_current_step(i + 1)
    
    def meta_train(self, step):
        raise NotImplementedError()

    def test(self):
        results = {}
        assert self.args.optim.evaluation_task_steps and len(self.args.optim.evaluation_task_steps) > 0, 'Include at least one evaluation step'
        evaluation_runs = sorted(self.args.optim.evaluation_task_steps)
        if self.args.train.test_only:
            logger.info(f'Evaluating for adaption steps: {evaluation_runs}')

        for degradation_idx, (degradation_name, test_dl) in enumerate(self.test_dataloader.items()):
            results.setdefault(degradation_name, {})
            for b_i, benchmark in enumerate(self.args.data.data_test): 
                benchmark_size = len(test_dl[b_i])
                logger.debug(f'Currently testing for {degradation_name}:{benchmark} - size is {benchmark_size}')           
                results[degradation_name].setdefault(benchmark, {})

                if self.max_evaluation_count is not None:
                    stop_counter = 0
                    benchmark_size = self.max_evaluation_count

                for lr_son, lr_dad, lr, hr, fn, k in test_dl[b_i]:
                    logger.debug(f'Test task: kernel: {k.shape if k[0] != "null" else k}, fn: {fn}')
                    
                    learners = {}
                    for model_typ, model in self.meta_models.items():
                        learners[model_typ] = model.clone()

                    if self.gan and self.kernelgan_crop:
                        downsampling_task_steps = self.models['downsampling'].optim_args.task_adapt_steps
                        g_indices = tutils.create_indices(lr_dad, iterations=max(downsampling_task_steps, evaluation_runs[-1]),\
                                                     crop_size=self.dis_input_ps*2)
                        d_indices = None
                                                        
                    lr_son, lr_dad, lr, hr, k = tutils.to_device(*[lr_son, lr_dad, lr, hr, k], cpu=self.args.sys.cpu)
                    k = k.squeeze()

                    previous_step = 0

                    for er in evaluation_runs:      
                        self.test_adapt(learners, previous_step, er, data=(lr_son, lr_dad, lr, hr), 
                            test_degradation_idx=degradation_idx, degradation_name=degradation_name, fn=fn[0],
                            g_indices=g_indices, d_indices=d_indices, gt_kernel=k)
                        
                        psnr, ssim, k_psnr, k_cov, img = self._test(learners, data=(lr_son, lr_dad, lr, hr), \
                            gt_kernel=k)
                            
                        results[degradation_name][benchmark].setdefault(er, {})
                        if len(results[degradation_name][benchmark][er]) == 0:
                            results[degradation_name][benchmark][er] = defaultdict(float)

                        results[degradation_name][benchmark][er]['k_psnr'] += k_psnr / benchmark_size
                        results[degradation_name][benchmark][er]['k_cov'] += k_cov / benchmark_size                        
                        results[degradation_name][benchmark][er]['psnr'] += psnr / benchmark_size
                        results[degradation_name][benchmark][er]['ssim'] += ssim / benchmark_size
                        previous_step = er
                        if self.args.train.test_only:
                            log_msg = f'[{degradation_name}][{benchmark}][{fn[0]}][{er}]: {psnr}/{ssim}/{k_psnr}/{k_cov}'
                            logger.info(log_msg)

                    self._save_results(img, f'{degradation_name}_{fn[0]}')

                    if self.max_evaluation_count is not None:
                        stop_counter += 1
                        if stop_counter == self.max_evaluation_count:
                            break

        return results

    def _save_results(self, img, fn):
        if self.args.train.save_results:
            self.save_results(img, fn, self.scale)

    def test_adapt(self, learners, previous_step, current_step, data, **kwargs):
        raise NotImplementedError
    
    def _test(self, learners, data, patch_size=None, **kwargs):
        raise NotImplementedError
