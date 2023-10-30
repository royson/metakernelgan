import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import fkp
from trainer.trainer import Trainer
from trainer import kernelgan
import trainer.utils as tutils
import data.utils as dutils
from utils import cycle

import logging
logger = logging.getLogger(__name__)

class DatasetKernelGANTrainer(Trainer):
    '''
        Pretrain an initialization of KernelGAN using a dataset without meta-learning.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsgan_loss = kernelgan.GANLoss(self.dis_output_ps, batch_size=self.args.optim.task_batch_size, device=self.device)
        logger.info(f'[*] LSGAN Loss is enabled. Dis input size: {self.dis_input_ps}. Dis output size: {self.dis_output_ps}.') 

        assert 'downsampling' in self.models and 'discriminator' in self.models

        if not self.args.train.test_only:
            # Trains in x2 only. x4 kernel is derived from x2.
            assert self.scale == 2 

        for model_typ, model in self.models.items():
            opt_args = model.optim_args
            self.optimizers[model_typ] = tutils.get_optimizer(model, opt_args.opt, \
                                                opt_args.lr, opt_args.opt_args, verbose=False)

        self.degradation_operation = self.args.degradation_operations.train
        self.existing_step = 0
        self.steps = self.args.optim.steps

    def sample_task(self):
        kernel = dutils.get_downsampling_ops(self.degradation_operation, train=True, scale=self.scale)
        logger.debug(f'Train task: kernel: {kernel.shape if kernel is not None else kernel}')
        labels = next(self.train_iter)

        patch_lrs_dad_t = fkp.batch_degradation(labels, kernel, np.array([self.scale, self.scale]), self.args.data.img_noise, device=self.device)
        patch_lrs_son_t = fkp.batch_degradation(patch_lrs_dad_t, kernel, np.array([2, 2]), self.args.data.img_noise, device=self.device)

        if kernel is not None:
            kernel = torch.from_numpy(kernel).float()
        
        return patch_lrs_son_t, patch_lrs_dad_t, kernel

    def train(self):        
        for model in self.models.values():
            model.train()

        self.train_iter = iter(cycle(self.train_dataloader))
        assert self.existing_step < self.steps, 'Training is done.'

        logger.info(f'{self.steps - self.existing_step} steps left.')

        for i in range(self.existing_step, self.steps):
            gt_patch_son, patch_dad, gt_kernel = self.sample_task()
            gt_patch_son, patch_dad, gt_kernel = \
                    tutils.to_device(*[gt_patch_son, patch_dad, gt_kernel], cpu=self.args.sys.cpu)
            
            # update the generator
            self.optimizers['downsampling'].zero_grad() 
            patch_son = self.models['downsampling'](patch_dad)

            fake_d_in, real_d_in, coords = tutils.random_image_crop(patch_son, \
                            img_dad=patch_dad, scale=self.scale, patch_size=self.dis_input_ps)

            gen_error = self.compute_loss_generator(self.models['discriminator'], fake_i=fake_d_in)
            # add kernelgan reg losses
            reg_error = self.compute_kernel_reg(self.models['downsampling'])
            gen_error += reg_error

            k_error = self.compute_kernel_loss(self.models['downsampling'], gt_kernel)
            gen_error += k_error

            gen_error.backward()
            self.optimizers['downsampling'].step()

            # update the discriminator
            self.optimizers['discriminator'].zero_grad() 

            patch_son = self.models['downsampling'](patch_dad)
            patch_son = patch_son.detach()
            g_out = tutils.get_patch_using_coordinates(patch_son, coords, patch_size=self.dis_input_ps)

            dis_error = self.compute_loss_discriminator(self.models['discriminator'], real_i=real_d_in, fake_i=g_out)
            dis_error.backward()
            self.optimizers['discriminator'].step()

            log_msg = f'[Step {i+1}] Gen error: {gen_error.item()}, Dis error {dis_error.item()}'            
            logger.debug(log_msg)

            if (i + 1) % self.args.optim.save_every == 0:
                logger.info(log_msg)
                self.save_ckp(i + 1, models=self.models)

            self.log_current_step(i + 1)

    def test(self):
        raise NotImplementedError()
    