import torch

import sys
import yaml
import os
from log import Checkpoint
from utils import AttrDict, args_backward_compatibility, attr_to_dict, merge_dict, \
                    PrettySafeLoader, compute_mean_and_sd
from data import get_dataloader
from model import get_models
from importlib import import_module
from pprint import pformat

import logging
logger = logging.getLogger(__name__)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

def read_yaml(filepath):
    with open(filepath, 'r') as stream:
        try:
            return yaml.load(stream, Loader=PrettySafeLoader)
        except yaml.YAMLError as exc:
            logger.error(exc)
            return {}

def get_args():
    args = {}
    basepath = os.path.dirname(__file__)
    args = merge_dict(args, read_yaml(os.path.join(basepath, 'configs', 'default.yaml')))
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.endswith('.yaml'):
                args = merge_dict(args, read_yaml(arg))
            elif len(arg.split('=')) == 2:
                args = merge_dict(args, attr_to_dict(arg))
            else:
                logger.warning(f'unrecognizable argument: {arg}')
    args = merge_dict(args, read_yaml(os.path.join(basepath, 'configs', 'env.yaml')))

    return args
    
if __name__ == '__main__':
    dargs = get_args()

    if dargs['sys']['log']:
        ckp = Checkpoint(dargs)
    else:
        ckp = None

    args = AttrDict(dargs)
    args_backward_compatibility(args)
    logger.info(f'Command ran: {" ".join(sys.argv)}')
    logger.info(pformat(args))
    
    if args.run:
        torch.manual_seed(args.sys.seed)

        models = get_models(args)
        dataloaders = get_dataloader(args)
        module_name, class_name = args.trainer.rsplit(".", 1)
        trainer_cls = getattr(import_module(module_name), class_name)
        trainer = trainer_cls(args, models, dataloaders, ckp)

        if not args.train.test_only:
            trainer.train()
        else:
            all_results = {}
            for test_idx in range(args.train.no_of_tests):
                results = trainer.test()
                logger.debug(results)
                for degradation_name, ks in results.items():
                    logger.info(f'Test Case: {degradation_name}')
                    for k, ers in ks.items():
                        for er, res in ers.items():
                            log_msg = f'[{k}: Adaption Steps {er}]: {round(res["psnr"], 2)}'
                            log_msg += f'/{round(res["ssim"], 4)}'
                            log_msg += f'/{round(res["k_psnr"], 2)}'
                            log_msg += f'/{round(res["k_cov"], 2)}'
                            logger.info(log_msg)
                                    
                all_results[test_idx] = results
            
            if len(all_results) > 1:
                metrics = ['psnr', 'ssim', 'k_psnr', 'k_cov']
                results = compute_mean_and_sd(all_results, metrics)
                logger.info('[*********] Overall Result [*********]')
                for degradation_name, ks in results.items():
                    logger.info(f'Test Case: {degradation_name}')
                    for k, ers in ks.items():
                        for er, res in ers.items():
                            log_msg = f'[{k}: Adaption Steps {er}]: {round(res["psnr"][0], 2)}'
                            log_msg += f'/{round(res["ssim"][0], 4)}'
                            log_msg += f'/{round(res["k_psnr"][0], 2)}'
                            log_msg += f'/{round(res["k_cov"][0], 2)}'
                            logger.info(log_msg)
            else:
                results = all_results[0] 
            
            if ckp is not None:
                ckp.dump_results(results)
        
