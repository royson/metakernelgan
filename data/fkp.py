'''
Credits: https://github.com/JingyunLiang/FKP
'''

# Generate random Gaussian kernels and downscale images
import sys
import numpy as np
from scipy.ndimage import filters, measurements, interpolation
import os
import torch
import torch.nn.functional as F

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

# Function for centering a kernel
def kernel_shift(kernel, sf):
    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The idea kernel center
    # for image blurred by filters.correlate
    # wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))
    # for image blurred by F.conv2d. They are the same after kernel.flip([0,1])
    wanted_center_of_mass = (np.array(kernel.shape) - sf) / 2.

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


# Function for calculating the X4 kernel from the X2 kernel, used in KernelGAN
def analytic_kernel(k):
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


# Function for generating one fixed kernel
def gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise):
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    return kernel


# Function for generating one random kernel
def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise)

    return kernel

# Function for degrading multiple images: input img is torch, output img is torch, kernel is numpy
def batch_degradation(inputs, kernel, scale_factor, noise_im, device=torch.device('cuda')):
    # preprocess image and kernel
    with torch.no_grad():
        t_kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
        b,c,h,w = inputs.size()
        inputs = inputs.reshape(b*c,h,w).unsqueeze(1).to(device)
        inputs = F.pad(inputs, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2),
                    mode='circular')
        outputs = F.conv2d(inputs, t_kernel)
        outputs = outputs.reshape(b,c,h,w)
        outputs = outputs[..., ::scale_factor[0], ::scale_factor[1]]
        outputs += torch.from_numpy(np.random.normal(0, np.random.uniform(0, noise_im), list(outputs.shape))).to(device)
    return outputs

# Function for degrading one image
def degradation(input, kernel, scale_factor, noise_im, device=torch.device('cuda')):
    # preprocess image and kernel
    input = torch.from_numpy(input).type(torch.FloatTensor).to(device).unsqueeze(0).permute(3, 0, 1, 2)
    input = F.pad(input, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2),
                  mode='circular')
    kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    # blur
    output = F.conv2d(input, kernel)
    output = output.permute(2, 3, 0, 1).squeeze(3).cpu().numpy()
    # down-sample
    output = output[::scale_factor[0], ::scale_factor[1], :]
    # add AWGN noise
    output += np.random.normal(0, np.random.uniform(0, noise_im), output.shape)
    return output
