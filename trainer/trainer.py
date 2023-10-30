import torch
import torch.nn as nn
import numpy as np
from trainer import utils as tutils
from data import utils as dutils
from data import fkp
from trainer import kernelgan
from pprint import pformat

import logging
logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, args, models, dataloaders, ckp):
        self.args = args
        self.scale = args.train.scale
        self.ckp = ckp
        self.train_dataloader, self.test_dataloader = dataloaders
        self.models = models
        
        ### Optimizers and Schedulers are defined in trainers
        self.optimizers = {}
        self.schedulers = {}

        self.device = torch.device('cpu' if args.sys.cpu else 'cuda')

        self.existing_step = 0
        self.current_loss = []
        
        ### Losses
        self.l1_loss = nn.L1Loss()
        self.l1_loss_reduction_sum = nn.L1Loss(reduction='sum')
        self.mse_loss = nn.MSELoss()
        self.gt_son_loss = self.args.optim.gt_son_loss
        
        ### Max no. of test images to evaluate in a dataset. Default is None (All)
        self.max_evaluation_count = self.args.data.max_evaluation_count 

        ### Initializing GAN parameters
        self.gan = False
        if 'discriminator' in self.models:
            assert 'downsampling' in self.models
            self.gan = True
            self.kernelgan_reg = False
            self.bicubic_loss = False
            self.stop_bicubic_check = True
            self.gan_noise = self.args.optim.lsgan_loss.noise
            self.kernel_size = self.models['downsampling'].kernel_size

            if self.gan_noise:
                logger.info(f'[*] Noise is added to inputs to the discriminator')

            self.dis_input_ps = self.args.optim.lsgan_loss.dis_input_patch_size
            with torch.no_grad():
                self.dis_output_ps = self.models['discriminator'](torch.ones(1, 3, self.dis_input_ps, self.dis_input_ps).to(self.device)).size(-1)   

            reg_losses = [kernelgan.SumOfWeightsLoss().to(self.device),\
                        kernelgan.BoundariesLoss(k_size=self.kernel_size, device=self.device).to(self.device),\
                        kernelgan.CentralizedLoss(k_size=self.kernel_size, scale_factor=(1/self.scale), device=self.device).to(self.device),\
                        kernelgan.SparsityLoss().to(self.device)]

            if self.args.optim.bicubic_loss:
                self.bicubic_stop = self.args.optim.bicubic_loss.bicubic_stop
                self.bicubic_loss = [kernelgan.DownScaleLoss(scale_factor=(1/self.scale), device=self.device).to(self.device), self.args.optim.kernelgan_reg[-1]]
                logger.info(f"[*] Bicubic Kernel Regularization Loss is enabled: {self.bicubic_loss}. Stopping after bicubic loss hits {self.bicubic_stop}.")
            
            self.kernelgan_reg = {}
            for hp, reg in zip(self.args.optim.kernelgan_reg, reg_losses):
                self.kernelgan_reg[reg._get_name()] = [reg, hp]
            
            self.reset_bicubic_loss()
                
            logger.info(f"[*] KernelGAN's Kernel Regularization Losses are enabled: {pformat(self.kernelgan_reg)}.")

            self.kernel_loss = self.args.optim.kernel_loss
            logger.info(f'[*] Training with magnitude of ground truth kernel. Loss weight: {self.kernel_loss}')
            self.kernelgan_crop = self.args.data.kernelgan_crop
            logger.info(f'[*] Gradient Magnitude Cropping: {self.kernelgan_crop}')

    def reset_bicubic_loss(self):
        if self.bicubic_loss:
            self.kernelgan_reg['CentralizedLoss'][1] = 0.0
            self.kernelgan_reg['SparsityLoss'][1] = 0.0
            self.stop_bicubic_check = False
                
            logger.debug(f"[*] KernelGAN's Kernel Regularization Losses have been reset: {pformat(self.kernelgan_reg)}.")

    def load_previous_ckp(self, models=None):
        if self.ckp is not None:
            if models is None:
                models = self.models
            
            self.existing_step, self.current_loss \
                = self.ckp.load_checkpoint(models, self.optimizers, self.schedulers) 
            
            if self.existing_step > 0:
                logger.info('Resuming training.')

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def save_ckp(self, step, models=None):
        if self.ckp is not None:
            if models is None:
                models = self.models
            
            self.ckp.save_checkpoint(step, self.current_loss, 
                models, self.optimizers, self.schedulers)

    def log_current_step(self, step):
        if self.ckp is not None:
            self.ckp.write_step(step)

    def save_kernel(self, kernel, fn):
        if self.ckp is not None:
            self.ckp.save_kernel(kernel, f'est_{fn}.mat')

    def save_results(self, img, fn, scale):
        if self.ckp is not None:
            self.ckp.save_results(img, f'x{scale}_{fn}')

    def dump_intermediate_results(self, img, fn, step):
        if self.ckp is not None:
            self.ckp.dump_intermediate_results(img, f'{fn}_{step}.png')

    def compute_loss_generator(self, model, fake_i):
        d_pred_fake = model(fake_i)
        gen_error = self.lsgan_loss(d_pred_fake, is_d_input_real=True)
        logger.debug(f'Gen loss: {gen_error.item()}')
        return gen_error
        
    def compute_loss_discriminator(self, model, real_i, fake_i):
        if self.gan_noise:
            real_i = real_i + torch.randn_like(real_i) / 255.
            fake_i = fake_i + torch.randn_like(fake_i) / 255.
        
        d_pred_real = model(real_i)
        d_pred_fake = model(fake_i)
        fake_error = self.lsgan_loss(d_pred_real, is_d_input_real=True)
        real_error = self.lsgan_loss(d_pred_fake, is_d_input_real=False)

        logger.debug(f'Dis real loss: {fake_error.item()}. Dis fake loss: {real_error.item()}')
        return (fake_error + real_error) * 0.5
    
    def compute_kernel_reg(self, generator, enable_bicubic_loss=False, g_input=None, g_output=None):
        loss = 0
        if self.kernelgan_reg:
            k = kernelgan.calc_curr_k(generator, estimated_kernel_size=self.kernel_size, cpu=self.args.sys.cpu)
            for name, (reg, hp) in self.kernelgan_reg.items():
                loss += hp * reg(k)
            if self.bicubic_loss and enable_bicubic_loss and not self.stop_bicubic_check:
                reg, hp = self.bicubic_loss
                bicubic_error =  hp * reg(g_input=g_input, g_output=g_output)
                loss += bicubic_error
                if bicubic_error.item() < self.bicubic_stop:
                    self.stop_bicubic_check = True
                    self.kernelgan_reg['CentralizedLoss'][1] = self.args.optim.kernelgan_reg[2]
                    self.kernelgan_reg['SparsityLoss'][1] = self.args.optim.kernelgan_reg[3]
                    logger.debug(f'Bicubic kernel achieved. New reg: {pformat(self.kernelgan_reg)}')
        return loss

    def kernel_covariance(self, kernel):
        # expects a torch blur kernel [h,w] and returns covariance matrix
        assert len(kernel.size()) == 2
        ksize = kernel.size(0)
        
        indices = np.indices([ksize, ksize])
        row_indices = torch.from_numpy(indices[0]).to(self.device)
        col_indices = torch.from_numpy(indices[1]).to(self.device)
        
        row_indices_mean = torch.sum(row_indices * kernel)
        col_indices_mean = torch.sum(col_indices * kernel)

        a = torch.sum((col_indices * col_indices) * kernel) - ((col_indices_mean ** 2))
        b = torch.sum((row_indices * row_indices) * kernel) - ((row_indices_mean ** 2))
        c = torch.sum((row_indices * col_indices) * kernel) - ((row_indices_mean * col_indices_mean))
        covar = torch.Tensor([[a, c], [c, b]]).to(self.device)
        
        return covar

    def compute_kernel_loss(self, model, gt_k):
        est_k = kernelgan.calc_curr_k(model, estimated_kernel_size=self.kernel_size, cpu=self.args.sys.cpu)
        assert est_k.shape == gt_k.shape

        loss = self.kernel_loss * self.l1_loss_reduction_sum(est_k, gt_k)
        logger.debug(f'Kernel L1 Loss: {loss}')
        return loss

    def process_and_save_kernel(self, model, degradation_name, fn, step):
        with torch.no_grad():
            est_k = kernelgan.calc_curr_k(model, estimated_kernel_size=self.kernel_size, cpu=self.args.sys.cpu)
            est_k = est_k.cpu().float().numpy()
            if self.scale == 4:
                est_k = kernelgan.analytic_kernel(est_k)
                est_k = fkp.kernel_shift(est_k, 4)
        self.save_kernel(est_k, f'{degradation_name}_{fn}_{step}')
