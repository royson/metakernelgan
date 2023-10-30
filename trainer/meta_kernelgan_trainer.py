import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import fkp
from trainer.meta_trainer import MetaTrainer
from trainer import kernelgan
import trainer.utils as tutils
import data.utils as dutils

import logging
logger = logging.getLogger(__name__)

class MetaKernelGANTrainer(MetaTrainer):
    '''
        Meta-learning for a variant of KernelGAN
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 'downsampling' in self.models and 'discriminator' in self.models
        assert self.lsgan_loss is not None
        
        if not self.args.train.test_only:
            # Trains in x2 only. x4 kernel is derived from x2.
            assert self.scale == 2 
            if self.args.optim.validate_steps:
                logger.info(f'[*] Validating {self.args.optim.validate_steps} steps during meta-train.')

        if self.args.train.evaluate_non_blind:
            from model.usrnet import USRNet
            usrnet = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                        nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
            usrnet.load_state_dict(torch.load(self.args.train.evaluate_non_blind.path), strict=True)
            usrnet.eval()
            logger.info(f'*Non-blind SR: USRNet Loaded from {self.args.train.evaluate_non_blind.path}')
            for key, v in usrnet.named_parameters():
                v.requires_grad = False
            self.usrnet = usrnet.to(self.device)

    def sample_task(self):
        kernel = dutils.get_downsampling_ops(self.degradation_operation, train=True, scale=self.scale)
        logger.debug(f'Train task: kernel: {kernel.shape if kernel is not None else kernel}')
        labels = next(self.train_iter)

        patch_lrs_dad_t = fkp.batch_degradation(labels, kernel, np.array([self.scale, self.scale]), self.args.data.img_noise, device=self.device)
        patch_lrs_son_t = fkp.batch_degradation(patch_lrs_dad_t, kernel, np.array([2, 2]), self.args.data.img_noise, device=self.device)

        if kernel is not None:
            kernel = torch.from_numpy(kernel).float()
        
        return patch_lrs_son_t, patch_lrs_dad_t, kernel

    def _validate_meta_train(self, learners, patch_dad, gt_patch_son, gt_kernel):
        patch_son = learners['downsampling'](patch_dad)  
        v_lr_son_error = 0
        loss = 0
        
        if self.gt_son_loss:    
            v_lr_son_error += self.l1_loss(patch_son, gt_patch_son)
            logger.debug(f'L1 LR_son Loss: {v_lr_son_error.item()}')

        if self.outer_lsgan_loss:
            d_in, _, coords = tutils.random_image_crop(patch_son, patch_size=self.dis_input_ps)
            g_error = self.compute_loss_generator(learners['discriminator'], d_in)
            v_lr_son_error += g_error
            reg_error = self.compute_kernel_reg(learners['downsampling'], enable_bicubic_loss=True, g_input=patch_dad, g_output=patch_son)
            v_lr_son_error += reg_error
            logger.debug(f'Reg Error: {reg_error.item()}, Gen Error: {g_error.item()}')

        if self.kernel_loss:            
            k_error = self.compute_kernel_loss(learners['downsampling'], gt_kernel)
            v_lr_son_error += k_error
            logger.debug(f'Total Kernel Error: {k_error.item()}')

        logger.debug(f'Total Gen error: {v_lr_son_error.item()}')
        
        dis_error = 0
        if self.outer_lsgan_loss:
            g_out = tutils.get_patch_using_coordinates(patch_son.detach(), coords, patch_size=self.dis_input_ps)
            real_d_in = tutils.get_patch_using_coordinates(gt_patch_son, coords, patch_size=self.dis_input_ps)
            dis_error = self.compute_loss_discriminator(learners['discriminator'], real_i=real_d_in, fake_i=g_out)
            logger.debug(f'Dis error: {dis_error.item()}')
        
        return v_lr_son_error, dis_error

    def lr_decay(self, step, model, area):
        if hasattr(model, 'set_lr'):
            if type(model.optim_args.task_lr_decay) == list \
                and step in model.optim_args.task_lr_decay:
                model.set_lr(model.lr / 10.)

            elif type(model.optim_args.task_lr_decay) == dict or \
                dict in type(model.optim_args.task_lr_decay).__bases__:
                for max_area, task_lr_decay_steps in model.optim_args.task_lr_decay.items():
                    if (max_area == 'default' or area <= int(max_area)):
                        if step in task_lr_decay_steps:
                            model.set_lr(model.lr / 10.)
                        break


    def meta_train(self, step):
        loss = 0
        learners = {}

        for model_typ, model in self.meta_models.items():
            learners[model_typ] = model.clone()
        
        gt_patch_son, patch_dad, gt_kernel = self.sample_task()
        gt_patch_son, patch_dad, gt_kernel = \
                tutils.to_device(*[gt_patch_son, patch_dad, gt_kernel], cpu=self.args.sys.cpu)

        v_lr_son_error = []
        v_dis_error = []
        area = np.prod(patch_dad.shape[-2:])
        for adapt_step in range(learners['downsampling'].optim_args.task_adapt_steps):
            patch_son = learners['downsampling'](patch_dad)
            fake_d_in, real_d_in, coords = tutils.random_image_crop(patch_son, \
                            img_dad=patch_dad, scale=self.scale, patch_size=self.dis_input_ps)


            gen_error = self.compute_loss_generator(learners['discriminator'], fake_i=fake_d_in)
            # add reg losses
            reg_error = self.compute_kernel_reg(learners['downsampling'])
            gen_error += reg_error

            self.lr_decay(adapt_step, learners['downsampling'], area)
            learners['downsampling'].adapt(gen_error, first_order=learners['downsampling'].optim_args.first_order)


            # update the discriminator
            patch_son = learners['downsampling'](patch_dad)
            patch_son = patch_son.detach()

            g_out = tutils.get_patch_using_coordinates(patch_son, coords, patch_size=self.dis_input_ps)
            
            dis_error = self.compute_loss_discriminator(learners['discriminator'], real_i=real_d_in, fake_i=g_out)

            learners['discriminator'].adapt(dis_error, first_order=learners['discriminator'].optim_args.first_order)

            if self.args.optim.validate_steps and (adapt_step + 1) in self.args.optim.validate_steps:
                logger.debug(f'Validating at step {adapt_step}')
                g_er, d_er = self._validate_meta_train(learners, patch_dad, gt_patch_son, gt_kernel)
                v_lr_son_error.append(g_er)
                v_dis_error.append(d_er)

        if not self.args.optim.validate_steps:
            v_lr_son_error, v_dis_error = self._validate_meta_train(learners, patch_dad, gt_patch_son, gt_kernel)
        else:
            loss_weights = self._get_loss_weights(step).to(self.device)
            v_lr_son_error = torch.sum(loss_weights * torch.stack(v_lr_son_error).squeeze())
            if self.outer_lsgan_loss:
                v_dis_error = torch.sum(loss_weights * torch.stack(v_dis_error).squeeze())

        # outer-loop
        loss += v_lr_son_error.item()
        v_lr_son_error.backward(retain_graph=not learners['discriminator'].optim_args.first_order)
        if not learners['downsampling'].optim_args.first_order and 'discriminator' in learners:
            self.optimizers['discriminator'].zero_grad()

        if self.outer_lsgan_loss:
            loss += v_dis_error.item()
            self.optimizers['discriminator'].zero_grad()
            v_dis_error.backward()

        for model_typ, model in self.meta_models.items():
            if model.optim_args.task_opt.upper() == 'ADAM': 
                # copy grad to model
                for model_p, clone_p in zip(model.parameters(), learners[model_typ].parameters()):
                    if model_p.requires_grad:
                        model_p.grad = clone_p.grad.clone()
                        
            if model.optim_args.task_gradient_clip is not None and model.optim_args.task_gradient_clip > 0:
                nn.utils.clip_grad_value_(model.parameters(), model.optim_args.task_gradient_clip)
        
        return loss
       
    def test_adapt(self, learners, previous_step, current_step, data, fn, g_indices, d_indices, \
                    degradation_name=None, gt_kernel=None, **kwargs):
        _, lr, _, _ = data 
        area = np.prod(lr.shape[-2:])
        step = -1
        for step in range(previous_step, current_step):               
            if type(self.args.train.save_kernels) == list and step in self.args.train.save_kernels:
                if step == 0:
                    gt_k = gt_kernel.cpu().float().numpy()
                    self.save_kernel(gt_k, f'{degradation_name}_{fn}_GT')
                self.process_and_save_kernel(learners['downsampling'], degradation_name, fn, step)

            patch_lr = tutils.kernelgan_lr_crop(lr, g_indices[step], self.dis_input_ps * 2)
            fake_d_in = learners['downsampling'](patch_lr)

            gen_error = self.compute_loss_generator(learners['discriminator'], fake_i=fake_d_in)
            logger.debug(f'GEN ERROR without reg: {gen_error}')
            gen_error += self.compute_kernel_reg(learners['downsampling'])

            self.lr_decay(step, learners['downsampling'], area)

                
            learners['downsampling'].adapt(gen_error, first_order=True)

            self.lr_decay(step, learners['discriminator'], area)


            # update the discriminator
            g_out = learners['downsampling'](patch_lr)
            g_out = g_out.detach()
            real_d_in = torch.cat([real_p[:,:self.dis_input_ps,:self.dis_input_ps].unsqueeze(0) for real_p in patch_lr],dim=0)

            dis_error = self.compute_loss_discriminator(learners['discriminator'], real_i=real_d_in, fake_i=g_out)

            learners['discriminator'].adapt(dis_error, first_order=True)

            logger.debug(f'GEN error: {gen_error.item()}. DIS error: {dis_error.item()}')

        if type(self.args.train.save_kernels) == list and step+1 in self.args.train.save_kernels:
            self.process_and_save_kernel(learners['downsampling'], degradation_name, fn, step+1)

    def _test(self, learners, data, gt_kernel):
        k = kernelgan.calc_curr_k(learners['downsampling'], estimated_kernel_size=self.kernel_size, cpu=self.args.sys.cpu)

        if self.scale == 4:
            k = k.detach().cpu().float().numpy()
            k = kernelgan.analytic_kernel(k)
            k = fkp.kernel_shift(k, 4)
            k = torch.from_numpy(k).float().to(self.device)            
        assert k.shape == gt_kernel.shape

        k_psnr = tutils.calc_psnr(k, gt_kernel, is_kernel=True) 
        with torch.no_grad():
            k_cov = self.l1_loss_reduction_sum(self.kernel_covariance(k), self.kernel_covariance(gt_kernel)).item()

        psnr = 0.0
        ssim = 0.0
        sr = None
        if self.args.train.evaluate_non_blind:
            _, _, lr, hr = data
            if self.args.train.evaluate_non_blind.model.lower() == 'usrnet':
                with torch.no_grad():
                    sr = self.usrnet(lr, torch.flip(k[None, None, :,:], [2, 3]), 4 if self.scale == 4 else 2,
                        (10 if self.args.data.real else 0) / 255 * torch.ones([1, 1, 1, 1]).to(self.device))
            else:
                sr = self.mzsr.run_adaption(lr, k, steps=100)

            sr, hr = tutils.post_process(*[sr, hr], preprocess=self.args.data.preprocess)
            
            sr_y, hr_y = tutils.rgb2y(*[sr, hr])

            psnr = tutils.calc_psnr(sr_y, hr_y, self.scale) 
            ssim = tutils.calc_ssim(sr_y, hr_y, self.scale) 
                    
        return psnr, ssim, k_psnr, k_cov, sr

    def _get_loss_weights(self, step):
        # adopted from MZSR
        # https://github.com/JWSoh/MZSR/blob/0e21434517da28a67cf0759e4af1544702c1b225/train.py#L210
        task_iter = len(self.args.optim.validate_steps)
        loss_weights = torch.ones(task_iter) * (1.0/task_iter)
        decay_rate = 1.0 / task_iter / (10000 / 3)
        min_value= 0.03 / task_iter

        loss_weights_pre = torch.maximum(loss_weights[:-1] - (step * decay_rate), torch.ones(loss_weights[:-1].size()) * min_value)
        loss_weight_cur = torch.minimum(loss_weights[-1] + (step * ((task_iter - 1) * decay_rate)), torch.ones(1) * (1.0 - ((task_iter - 1) * min_value)))
        loss_weights = torch.cat((loss_weights_pre, loss_weight_cur), dim=0)
        return loss_weights

    