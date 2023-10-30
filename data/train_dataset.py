import os
import torch
import glob
import lmdb
import numpy as np

import imageio

from utils import cycle
import data.utils as dutils
from torch.utils.data import Dataset, DataLoader
from trainer import kernelgan

import pyarrow as pa

import logging
logger = logging.getLogger(__name__)

class DIV2KDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.train.scale
        self.train_size = args.train.patch_size

        if args.data.data_folder is not None:
            self.data_folder = args.data.data_folder
        else:
            # default data_dir
            self.data_folder = 'train_HR'
                    
        self.data_dir = os.path.join(args.data.data_dir, args.data.data_train, self.data_folder)
        assert os.path.exists(self.data_dir), f'{self.data_dir} is not found.'
        self._create_lmdb()        
        
        self.crop_using_grad = self.args.data.kernelgan_crop and self.args.data.kernelgan_crop.train 
        if self.crop_using_grad:
            self.data_pmaps = None
            self._create_grad_lmdb()
            
        # initialize keys
        self._init_lmdb()
        self.env = None
        self.pmap_env = None
        
        logger.info(f'[*] Train data directory: {self.data_dir}. No. of training patches: {len(self.keys)}. \
            Gradient maps: {self.crop_using_grad}. ')

    def _create_lmdb(self):
        if not os.path.exists(os.path.join(self.data_dir, 'data.mdb')) or \
                not os.path.exists(os.path.join(self.data_dir, 'lock.mdb')):
            logger.info(f'LMDB not found in {self.data_dir}. Regenerating from source: {self.data_dir}')
            dutils.create_lmdb_imgs(self.data_dir, lmdb_dir=self.data_dir)

    def _create_grad_lmdb(self):
        grad_folder = os.path.join(self.args.data.data_dir, self.args.data.data_train, f'{self.data_folder}_g{self.train_size}')
        if not os.path.exists(grad_folder):
            os.makedirs(grad_folder)
        
        if not os.path.exists(os.path.join(grad_folder, 'data.mdb')) or \
                not os.path.exists(os.path.join(grad_folder, 'lock.mdb')):
            source_path = os.path.join(self.args.data.data_dir, self.args.data.data_train, self.data_folder)
            logger.info(f'LMDB prob. maps not found in {grad_folder}. Regenerating from source: {source_path}')
            dutils.create_lmdb_gradient_maps(source_path, lmdb_folder=grad_folder,\
                 crop_size=self.train_size, scale=self.scale)

        self.data_pmaps = grad_folder

    def _init_lmdb(self):
        self.env = lmdb.open(self.data_dir, subdir=os.path.isdir(self.data_dir), 
                        readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.keys = pa.deserialize(txn.get(b'__keys__'))
        
        if self.crop_using_grad:
            self.pmap_env = lmdb.open(self.data_pmaps, subdir=os.path.isdir(self.data_pmaps), 
                            readonly=True, lock=False, readahead=False, meminit=False)
            with self.pmap_env.begin(write=False) as txn:             
                assertion_keys = pa.deserialize(txn.get(b'__keys__'))
                assert self.keys == assertion_keys

    def _load_from_buffer(self, idx):
        env = self.env
        pmap = None

        with env.begin(write=False) as txn:
            bflow = txn.get(self.keys[idx])
        
        img = pa.deserialize(bflow).astype(np.float32)

        if self.crop_using_grad:
            pmap_env = self.pmap_env
            with pmap_env.begin(write=False) as txn:
                map_bflow = txn.get(u'{}'.format(f'{idx}_pmap').encode('ascii'))
            pmap = pa.deserialize(map_bflow).astype(np.float16)

        return img, pmap

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_lmdb()

        patch_hr, pmap = self._load_from_buffer(idx)
        patch_hr = dutils.modcrop(patch_hr, self.scale) 

        if self.crop_using_grad:
            pmap /= pmap.sum() # deal w floating point roundoff error
            center = np.random.choice(a=len(pmap), size=1, p=pmap)[0]
            top, left = kernelgan.get_top_left(center, size=self.train_size, img_shape=patch_hr.shape)
            patch_hr = patch_hr[top:top + self.train_size, left:left + self.train_size, :]
        else:
            patch_hr = dutils.get_patch(patch_hr, patch_size=self.train_size)

        patch_hr = dutils.augment(patch_hr)
        patch_hr = dutils.to_tensor(patch_hr)[0]

        if self.args.data.preprocess:
            patch_hr = dutils.preprocess(patch_hr)[0]
        
        return patch_hr
