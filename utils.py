import os
import re
import ast
import yaml
import copy
import numpy as np

class PrettySafeLoader(yaml.SafeLoader):
    '''
    Allows yaml to load tuples. Credits to Matt Anderson. See: 
    https://stackoverflow.com/questions/9169025/how-can-i-add-a-python-tuple-to-a-yaml-file-using-pyyaml
    '''
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

class AttrDict(dict):
    def __init__(self, d={}):
        super(AttrDict, self).__init__()
        for k, v in d.items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
        if isinstance(v, dict):
            v = AttrDict(v)
        super(AttrDict, self).__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return self.__getitem__(k)
        except KeyError:
            raise AttributeError(k)

    __setattr__ = __setitem__

def attr_to_dict(attr):
    '''
        Transforms attr string to nested dict
    '''
    nested_k, v = attr.split('=')
    ks = nested_k.split('.')
    d = {}
    ref = d
    while len(ks) > 1:
        k = ks.pop(0)
        ref[k] = {}
        ref = ref[k]

    ref[ks.pop()] = assign_numeric_type(v)

    return d
 
def assign_numeric_type(v):
    if re.match(r'^-?\d+(?:\.\d+)$', v) is not None:
        return float(v)
    elif re.match(r'^-?\d+$', v) is not None:
        return int(v)
    elif re.match(r'^range\(-?\d+,-?\d+,-?\d+\)$', v) is not None:
        r_nos = v.split('range(')[-1][:-1].split(',')
        return list(range(int(r_nos[0]), int(r_nos[1]), int(r_nos[2])))
    elif v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    elif v.lower() == 'null':
        return None
    else: 
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError) as e:
            return v

def merge_dict(a, b):
    '''
        merge dictionary b into dictionary a
    '''
    assert isinstance(a, dict) and isinstance(b, dict)
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def args_backward_compatibility(args):
    # to set default values of missing or old arguments defined in previous versions of this framework

    if not hasattr(args.train, 'no_of_tests'):
        args.train.no_of_tests = 1
    
    # for data 
    if not hasattr(args.data, 'data_folder'):
        args.data.data_folder = None

    if not hasattr(args.data, 'kernelgan_crop'):
        args.data.kernelgan_crop = False

    if not hasattr(args.data, 'real'):
        args.data.real = False

    if not hasattr(args.data, 'test_only_images'):
        args.data.test_only_images = None

    if hasattr(args.data, 'custom_data_path') and args.data.custom_data_path is not None:
        args.data.data_test = ['custom']

    if args.data.kernelgan_crop is True:
        args.data.kernelgan_crop = AttrDict({})
        args.data.kernelgan_crop.train = True
        args.data.kernelgan_crop.test = True
        args.data.kernelgan_crop.scale_crop = True        
    elif args.data.kernelgan_crop is False:
        args.data.kernelgan_crop = AttrDict({})
        args.data.kernelgan_crop.train = False
        args.data.kernelgan_crop.test = False
        args.data.kernelgan_crop.scale_crop = False
    else:
        if not hasattr(args.data.kernelgan_crop, 'train'):
            args.data.kernelgan_crop.train = False
        if not hasattr(args.data.kernelgan_crop, 'test'):
            args.data.kernelgan_crop.test = False
        if not hasattr(args.data.kernelgan_crop, 'scale_crop'):
            args.data.kernelgan_crop.scale_crop = False

    if not hasattr(args.data, 'max_evaluation_count'):
        args.data.max_evaluation_count = None

    if not hasattr(args.data, 'img_noise'):
        args.data.img_noise = 0.

    if not hasattr(args.optim, 'gt_son_loss'):
        args.optim.gt_son_loss = True

    if not hasattr(args.optim, 'validate_steps'):
        args.optim.validate_steps = [] 

    if args.optim.lsgan_loss:
        assert hasattr(args.model, 'discriminator'), 'LSGAN loss is specified but not a discriminator'
        assert hasattr(args.model, 'downsampling'), 'A Downscaling model must be present w a discriminator'
        assert hasattr(args.optim, 'kernelgan_reg') and args.optim.kernelgan_reg
        if not type(args.optim.kernelgan_reg) == list:
            args.optim.kernelgan_reg = [0.5, 0.0, 0, 0, 0.0]

        if not hasattr(args.optim.lsgan_loss, 'noise'):
            args.optim.lsgan_loss.noise = True        
            
        assert len(args.optim.kernelgan_reg) == 5
        if not hasattr(args.optim, 'bicubic_loss'):
            args.optim.bicubic_loss = False
        
        if not hasattr(args.train, 'evaluate_non_blind'):
            args.train.evaluate_non_blind = False
        
        if args.train.evaluate_non_blind:
            if args.train.evaluate_non_blind == True:
                args.train.evaluate_non_blind = AttrDict({})
                args.train.evaluate_non_blind.model = 'usrnet'
            assert hasattr(args.train.evaluate_non_blind, 'model') and args.train.evaluate_non_blind.model.lower() == 'usrnet'
            if not hasattr(args.train.evaluate_non_blind, 'path'):
                args.train.evaluate_non_blind.path = os.path.join(args.home_dir, 'model', 'usrnet_tiny.pth')
            assert os.path.exists(args.train.evaluate_non_blind.path)

        if not hasattr(args.optim, 'kernel_loss'):
            args.optim.kernel_loss = False

        if not hasattr(args.optim.lsgan_loss, 'dis_input_patch_size'):
            args.optim.lsgan_loss.dis_input_patch_size = 32

        if not hasattr(args.optim, 'outer_lsgan_loss'):
            args.optim.outer_lsgan_loss = False

def compute_mean_and_sd(all_results, metrics=['psnr', 'ssim']):
    assert len(all_results) > 1 
    result_summary = copy.deepcopy(all_results[0])
    del all_results[0]
    # compile results
    for d_idx in result_summary.keys():
        for result in all_results.values():
            for benchmark in result[d_idx].keys():
                for adapt_step in result[d_idx][benchmark].keys():
                    sum_list = result_summary[d_idx][benchmark][adapt_step]

                    for m in metrics:
                        if type(sum_list[m]) != list:
                            sum_list[m] = [sum_list[m]]
                        sum_list[m].append(result[d_idx][benchmark][adapt_step][m])
    # return each result in a tuple (mean, sd)
    for d_idx in result_summary.keys():
        for benchmark in result_summary[d_idx].keys():
            for adapt_step in result_summary[d_idx][benchmark].keys():
                sum_list = result_summary[d_idx][benchmark][adapt_step]
                for m in metrics:
                    sum_list[m] = (np.mean(sum_list[m]), np.std(sum_list[m]))

    return result_summary
                    
    
