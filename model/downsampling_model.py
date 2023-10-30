import torch
import torch.nn as nn
import torch.nn.functional as F
import model.utils as mutils

import logging
logger = logging.getLogger(__name__)

class DownsamplingModel(nn.Module):
    def __init__(self, args, chnls=64, kernel_size=15, **kwargs):
        super(DownsamplingModel, self).__init__()
        logger.info(f'[*] DownsamplingModel selected.')
        self.args = args

        self.downscale = nn.Sequential(nn.Conv2d(1, chnls, 7, bias=False),
                        nn.Conv2d(chnls, chnls, 3, bias=False),
                        nn.Conv2d(chnls, chnls, 3, bias=False),
                        nn.Conv2d(chnls, chnls, 1, bias=False),
                        nn.Conv2d(chnls, chnls, 1, bias=False),
                        nn.Conv2d(chnls, 1, 1, stride=2, bias=False))

        self.kernel_size = kernel_size
	
    def forward(self, x, **kwargs):
        b, c, h, w = x.size()
        x = x.reshape(b*c, 1, h, w)
        x = F.pad(x, (self.kernel_size//2, self.kernel_size//2, self.kernel_size//2, self.kernel_size//2), 'circular')
        x = self.downscale(x)
        return x.reshape(b, c, x.size(2), x.size(3))             
