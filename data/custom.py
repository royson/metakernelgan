import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import data.utils as dutils
from trainer.kernelgan import analytic_kernel
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, data_path):
        self.data_dir = data_path
        self.args = args
        self.ksize = self.args.model.downsampling.model_args.kernel_size
        
        self.scale = args.train.scale
        if self.scale == 4:
            k = np.ones((self.ksize, self.ksize))
            k = analytic_kernel(k)
            self.ksize = k.shape[0]
        import glob

        ext = ['png', 'jpg', 'jpeg']
        self.fn_lr = []
        [self.fn_lr.extend(globl.glob(osp.join(args.input_dir, '*.' + e))) for e in ext]
        self.fn_lr = sorted(self.fn_lr)
        assert self.fn_lr, f'No data found. Current data path is {self.data_dir}'
        
    def __len__(self):
        return len(self.fn_lr)
    
    def __getitem__(self, idx):
        f_lr = self.fn_lr[idx]
        img_lr = imageio.imread(f_lr).astype(np.float32)
        img_lr = dutils.set_rgb_channel(img_lr)
        img_lr_dad = np.copy(img_lr)
        img_lr_dad_t, img_lr_t = dutils.to_tensor(*[img_lr_dad, img_lr])

        if self.args.data.preprocess:
            img_lr_dad_t, img_lr_t = dutils.preprocess(*[img_lr_dad_t, img_lr_t])
        
        c,h,w = img_lr_t.size()

        # placeholder values for pipeline
        img_hr_t = torch.ones(c, h * self.scale, w * self.scale)
        img_lr_son_t = torch.ones(c, h // self.scale, w // self.scale)
        print(img_lr_son_t.size(), img_lr_dad_t.size(), img_lr_t.size(), img_hr_t.size(), f_lr, self.ksize)
        return img_lr_son_t, img_lr_dad_t, img_lr_t, img_hr_t, os.path.basename(f_lr), \
            torch.ones(self.ksize,self.ksize)
