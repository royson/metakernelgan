import random
import numpy as np
import torch
import torch.nn.functional as F
import os
import glob
import imageio
import lmdb
import numpy as np
import pyarrow as pa
import math

from data.fkp import gen_kernel_fixed, gen_kernel_random, modcrop
from scipy.io import loadmat
from scipy.ndimage import filters, measurements, interpolation
from math import pi
from io import BytesIO
from PIL import Image
from trainer import kernelgan
from tqdm import tqdm

def create_lmdb_imgs(img_dir, lmdb_dir):
    assert os.path.exists(img_dir)
    os.makedirs(lmdb_dir, exist_ok=True)
    write_freq = 20
    fn_hr = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    
    db = lmdb.open(lmdb_dir, subdir=True, 
            map_size=1099511627776,
            readonly=False,
            meminit=False,
            map_async=True)

    txn = db.begin(write=True)

    for idx, fn in enumerate(tqdm(fn_hr, ascii=True, desc='Generating LMDB')):
        img = imageio.imread(fn).astype(np.uint8)

        txn.put(u'{}'.format(idx).encode('ascii'), 
            pa.serialize(np.asarray(img)).to_buffer())
        
        if idx % write_freq == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(fn_hr))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pa.serialize(keys).to_buffer())
        
    db.sync()
    db.close()

def create_lmdb_gradient_maps(source_path, lmdb_folder,
        crop_size, scale, ext='*.png', write_freq=20):
    assert os.path.exists(source_path)
    fn_hr = sorted(glob.glob(os.path.join(source_path, ext)))

    db = lmdb.open(lmdb_folder, subdir=True, 
            map_size=1099511627776,
            readonly=False,
            meminit=False,
            map_async=True)

    txn = db.begin(write=True)

    for idx, fn in enumerate(tqdm(fn_hr, ascii=True, desc='Generating LMDB prob. maps')):
        img = imageio.imread(fn).astype(np.uint8)

        img = modcrop(img, scale)
        img_grad = kernelgan.create_gradient_map(img / 255.)
        prob_map = kernelgan.create_probability_map(img_grad, crop_size).astype(np.float16)
        txn.put(u'{}'.format(f'{idx}_pmap').encode('ascii'), 
            pa.serialize(np.asarray(prob_map)).to_buffer())

        if idx % write_freq == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(fn_hr))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pa.serialize(keys).to_buffer())
        
    db.sync()
    db.close()

def pad_to_min(img, min_size):
    _, _, h, w = img.size()
    pad_ver = 0
    pad_hor = 0
    if h < min_size:
        pad_ver = min_size - h
    if w < min_size:
        pad_hor = min_size - w

    return F.pad(img, (pad_hor // 2, pad_hor - (pad_hor // 2), pad_ver, pad_ver - (pad_ver // 2)), mode='circular')

def set_rgb_channel(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    
    if img.shape[2] == 1:
        img = np.concatenate([img] * 3, 2)

    return img

def to_tensor(*imgs):
    def _to_tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()

        return tensor

    return [_to_tensor(i) for i in imgs]

def preprocess(*imgs):
    def _preprocess(img):
        img = img.float().div(255.)

        return img

    return [_preprocess(i) for i in imgs]


def get_patch(img, patch_size):
    h, w = img.shape[:2]
    y = random.randrange(0, h - patch_size + 1)
    x = random.randrange(0, w - patch_size + 1)

    return img[y:y+patch_size, x:x+patch_size]

def augment(img):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    if hflip: img = img[:, ::-1, :]
    if vflip: img = img[::-1, :, :]
    if rot90: img = img.transpose(1, 0, 2)

    return img

def get_downsampling_ops(degradation_operation, train=True, scale=2, data_dir=None, fn=None):    
    kernel = degradation_operation.kernel
    
    if not hasattr(kernel, 'noise'):
        kernel.noise = 0.

    if not train:
        # fixed kernel
        if hasattr(kernel, 'kernel_folder') and kernel.kernel_folder is not None:
            img_kernel_path = os.path.join(data_dir, kernel.kernel_folder, f'{fn}.mat')
            assert os.path.exists(img_kernel_path), f'{img_kernel_path} does not exist.'
            kernel = loadmat(img_kernel_path)['Kernel']
        else:
            kernel = gen_kernel_fixed(k_size=np.array([kernel.ksize, kernel.ksize]), 
                            scale_factor=np.array([scale, scale]), lambda_1=kernel.l1, lambda_2=kernel.l2,
                            theta=kernel.theta, noise=kernel.noise)    
    else:
        # random kernel
        kernel = gen_kernel_random(k_size=np.array([kernel.ksize, kernel.ksize]), 
                        scale_factor=np.array([scale, scale]), min_var=kernel.lb, max_var=kernel.ub, 
                        noise_level=kernel.noise)
    return kernel