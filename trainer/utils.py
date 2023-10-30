import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from trainer import kernelgan
from data import utils as dutils

import logging
logger = logging.getLogger(__name__)

_ssim_filter = None

def get_optimizer(model, opt, lr, opt_args=None, verbose=True):
    if opt_args is None:
        opt_args = {}
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **opt_args)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, **opt_args)
    else:
        raise NotImplementedError()
    if verbose:
        logger.info(f'[*] {model.model_name} - Outer-loop Optimizer: {optimizer}')
    return optimizer

def create_indices(img, iterations, crop_size, interpolate_prob_map=False, scale=None):
    assert img.size(0) == 1
    np_img = img.detach().squeeze(0).permute(1, 2, 0).numpy() 
    img_grad = kernelgan.create_gradient_map(np_img)
    if interpolate_prob_map:
        assert scale is not None
        img_grad = kernelgan.nn_interpolation(img_grad, sf=scale)
    prob_map = kernelgan.create_probability_map(img_grad, crop_size)
    return np.random.choice(a=len(prob_map), size=iterations, p=prob_map)
     
def get_patch_using_coordinates(img, coordinates, patch_size=32):
    patches = None
    img = dutils.pad_to_min(img, patch_size)
    for h_i, w_i in coordinates:
        patch = img[:,:,h_i:h_i+patch_size, w_i:w_i+patch_size]
        if patches is None:
            patches = patch
        else:
            patches = torch.cat((patches, patch), dim=0)

    return patches

def kernelgan_lr_crop(img, center, crop_size):
    '''
    Crop LR image for G input using KG indices
    '''
    img = dutils.pad_to_min(img, crop_size)
    top, left = kernelgan.get_top_left(center=center, size=crop_size, img_shape=img.size()[2:])
    patch_dad = img[:,:,top:top + crop_size, left:left + crop_size]
    return patch_dad


def kernelgan_image_crop(img_dad, img_son, center, crop_size, patch_size=32, img_son_center=None):
    '''
        Sampling based on gradient magnitude. Either uses corresponding or given patch son. 
    '''
    _, _, h, w = img_dad.size()
    # img_dad = dutils.pad_to_min(img_dad, patch_size)
    top, left = kernelgan.get_top_left(center=center, size=crop_size, img_shape=img_dad.size()[2:])
    patch_dad = img_dad[:,:,top:top + patch_size, left:left + patch_size]
    if img_son_center is None:
        # corresponding patch
        h_i = top // 2 
        w_i = left // 2
    else:
        img_son = dutils.pad_to_min(img_son, patch_size)
        h_i, w_i = kernelgan.get_top_left(center=img_son_center, size=crop_size, img_shape=img_son.size()[2:])
    patch_son = img_son[:,:,h_i:h_i + patch_size, w_i:w_i + patch_size]
    return patch_son, patch_dad, [(h_i, w_i)]

def random_image_crop(img_son, img_dad=None, scale=None, samples=1, patch_size=32):
    '''
        Random sampling of patches from an image. Uses corresponding patch dad patch.
    '''
    _, _, h, w = img_son.size()
    patch_sons = None
    patch_dads = None

    if img_dad is not None:
        assert scale is not None

    h_is = random.sample(range(h-patch_size+1), samples)
    w_is = random.sample(range(w-patch_size+1), samples)
    patch_son_coords = list(zip(h_is, w_is))
    
    for h_i, w_i in patch_son_coords:
        if img_dad is not None:
            # corresponding patch
            up_h_i = h_i * scale
            up_w_i = w_i * scale
            patch_dad = img_dad[:,:,up_h_i:up_h_i+patch_size,up_w_i:up_w_i+patch_size]
            if patch_dads is None:
                patch_dads = patch_dad
            else:
                patch_dads = torch.cat((patch_dads, patch_dad), dim=0)
        patch_son = img_son[:,:,h_i:h_i+patch_size, w_i:w_i+patch_size]
        if patch_sons is None:
            patch_sons = patch_son
        else:
            patch_sons = torch.cat((patch_sons, patch_son), dim=0)
    
    return patch_sons, patch_dads, patch_son_coords

def check_state_dict(sd1, sd2):
    same = True
    for k in sd1.keys():
        sd2_k = k
        if k not in sd2:
            sd2_k = '.'.join(k.split('.')[1:])
        same = (torch.abs(torch.sum(sd1[k] - sd2[sd2_k]))) / (torch.numel(sd1[k])) < 1e-4
        if not same:
            logger.info(f'{k} is not the same!')
            return same
    return same

def get_meta_algorithm(args, algo, model, task_lr, first_order, task_opt, task_opt_args, gradient_clip=None, device='cuda'):
    assert algo in ['MAML'], 'Invalid meta-algorithm selected'
    if task_opt_args is None:
        task_opt_args = {}

    logger.info(f'[*] {algo} selected for {model._get_name()}! First order: {first_order}. Gradient clip: {gradient_clip}. task LR: {task_lr}. ')
    
    adapt_opt_sd = None
    if task_opt.upper() == 'ADAM':
        adapt_opt = torch.optim.Adam(model.parameters(), lr=task_lr, **task_opt_args)
        logger.info(f'[*] {model._get_name()} Inner-Loop Optimizer: {adapt_opt}')
        adapt_opt_sd = adapt_opt.state_dict()
    else:
        logger.info(f'[*] {model._get_name()} Inner-Loop Optimizer: {task_opt.upper()}')

    from algo.maml import MAMLWrapper
    meta_m = MAMLWrapper(model, lr=task_lr, gradient_clip=gradient_clip, first_order=first_order, \
                        adapt_opt_sd=adapt_opt_sd, adapt_opt_args=task_opt_args, allow_nograd=False)

    
    return meta_m
    
def to_device(*args, cpu=False):
    def _to_device(t):
        if t is None:
            return None
        return t.to(torch.device('cpu' if cpu else 'cuda'))

    return [_to_device(a) for a in args]

def rgb2y(*args):

    def _rgb2y(image):
        Kr = np.float64(0.299)
        Kb = np.float64(0.114)

        image = image.to(dtype=torch.float64) / 255
        gray_coeffs = image.new_tensor([Kr, 1-Kr-Kb, Kb]).view(1, 3, 1, 1)
        
        image = image.mul(gray_coeffs).sum(dim=1, keepdim=True)*np.float64(219)/255 + np.float64(16)/255
        
        assert image.size()[1] == 1
        return image * 255

    return [_rgb2y(a) for a in args]

def _prepare_ssim_filter(size=11, sigma=1.5):
    mesh = np.ndarray([size, size], dtype=np.float64)
    for x in range(0, size):
        for y in range(0, size):
            xx = int(x - (size-1)/2)
            yy = int(y - (size-1)/2)
            mesh[(x, y)] = math.exp(-float(xx**2 + yy**2) / float(2*sigma**2))

    mesh = mesh/mesh.sum()
    global _ssim_filter
    _ssim_filter = torch.tensor(mesh).view(1, 1, size, size)


def calc_psnr(sr_y, hr_y, scale=None, is_kernel=False):
    diff = sr_y - hr_y
    if not is_kernel:          
        shave = scale
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    if not is_kernel:
        return 10 * math.log10(255**2/mse)
    else:
        return 10 * math.log10(1.0/mse)

def calc_ssim(sr, hr, scale, rgb_range=255):
    global _ssim_filter
    if _ssim_filter is None:
        _prepare_ssim_filter()
        
    assert _ssim_filter is not None
    _ssim_filter = _ssim_filter.to(device=sr.device)

    shave = scale
    if shave > 0:
        sr = sr.to(dtype=torch.float64)[..., shave:-shave, shave:-shave]
        hr = hr.to(dtype=torch.float64)[..., shave:-shave, shave:-shave]

    K = sr.new_tensor([0.01, 0.03], dtype=torch.float64)
    C = K.mul(rgb_range).pow(2)

    mu1 = torch.nn.functional.conv2d(hr, _ssim_filter)
    mu2 = torch.nn.functional.conv2d(sr, _ssim_filter)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1.mul(mu2)

    sigma1_sq = torch.nn.functional.conv2d(hr.pow(2), _ssim_filter) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(sr.pow(2), _ssim_filter) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(hr.mul(sr), _ssim_filter) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C[0]).mul(2*sigma12 + C[1])).div((mu1_sq + mu2_sq + C[0]).mul(sigma1_sq + sigma2_sq + C[1]))
    return ssim_map.mean().item()


def post_process(*args, preprocess=False):
    def _post_process(img):
        if preprocess:
            img = img.mul(255)

        return img.clamp(0, 255).round().byte()
    
    return [_post_process(a) for a in args]
