'''
Discriminator from KernelGAN: https://github.com/sefibk/KernelGAN/blob/master/networks.py
'''

import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, args, ic=3, chnls=64, ks=7):
        super(Discriminator, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=ic, out_channels=chnls, kernel_size=ks, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, 6):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=chnls, out_channels=chnls, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(chnls),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=chnls, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)