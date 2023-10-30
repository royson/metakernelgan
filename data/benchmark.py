import os
import imageio
import data.utils as dutils
import glob

import numpy as np
import copy
import torch
from data import fkp
from torch.utils.data import Dataset

import pyarrow as pa

import logging
logger = logging.getLogger(__name__)

class BenchmarkDataset(Dataset):
    def __init__(self, args, name=None, degradation_operation=None, test=True):
        logger.debug(f'New Benchmark | name:{name}, do:{degradation_operation.name}, test:{test}')
        self.args = args
        self.device = torch.device('cpu' if args.sys.cpu else 'cuda')
        self.scale = args.train.scale
        self.name = name
        self.degradation_operation = degradation_operation
        self.test = test

        # Loading of HR images 
        self.hr_data_dir = os.path.join(args.data.data_dir, name, 'HR')
        self.fn_hr = sorted(glob.glob(os.path.join(self.hr_data_dir, '*.png')))
        assert self.fn_hr, f'No data found. Current HR data path is {self.hr_data_dir}'
        
        # Test only these images:
        if args.data.test_only_images is not None:
            self.fn_hr = [fn for fn in self.fn_hr if os.path.basename(fn) in args.data.test_only_images]

        self.length = len(self.fn_hr)

        logger.info(f'No. of test data ({self.name}): {self.length}. ')

        self.lr_son_path = None

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        f_hr = self.fn_hr[idx]
        
        img_hr = imageio.imread(f_hr).astype(np.float32)
        img_hr = dutils.set_rgb_channel(img_hr)
        img_hr = fkp.modcrop(img_hr, self.scale)
        kernel = dutils.get_downsampling_ops(self.degradation_operation, 
                                        train=False, 
                                        scale=self.scale, 
                                        data_dir=os.path.join(self.args.data.data_dir, self.name),
                                        fn=os.path.splitext(os.path.basename(f_hr))[0]
                                        )
                                        
        assert hasattr(self.degradation_operation, 'lr_folder'), 'Requires provided LR and kernel'
        lr_folder_path = os.path.join(self.args.data.data_dir, self.name, self.degradation_operation.lr_folder)
        assert os.path.exists(lr_folder_path), f'{lr_folder_path} not found.'

        img_lr = imageio.imread(os.path.join(lr_folder_path, os.path.basename(f_hr)), pilmode='RGB').astype(np.float32)
        img_lr_dad = np.copy(img_lr)
        img_lr_dad_t, img_lr_t, img_hr_t = dutils.to_tensor(*[img_lr_dad, img_lr, img_hr])
        img_lr_son_t = fkp.batch_degradation(img_lr_dad_t.unsqueeze(0), kernel, np.array([2, 2]), 0., device=torch.device('cpu')) 
        img_lr_son_t = dutils.pad_to_min(img_lr_son_t, min_size=self.args.optim.lsgan_loss.dis_input_patch_size).squeeze()

        if self.args.data.preprocess:
            img_lr_son_t, img_lr_dad_t, img_lr_t, img_hr_t = \
                dutils.preprocess(*[img_lr_son_t, img_lr_dad_t, img_lr_t, img_hr_t])

        if kernel is not None:
            kernel = torch.from_numpy(kernel).float()

        logger.debug(f"{os.path.basename(f_hr)}: {img_lr_son_t.size()}, {img_lr_dad_t.size()}, {img_lr_t.size()}, {img_hr_t.size()}, {kernel.size()}")
        return img_lr_son_t, img_lr_dad_t, img_lr_t, img_hr_t, os.path.basename(f_hr), kernel