import os
import yaml
import torch
import sys
import imageio
import copy
import pickle
import scipy.io as sio

from datetime import datetime

import logging
logger = logging.getLogger(__name__)

class Checkpoint():
    def __init__(self, args):
        if 'name' not in args:
            args['name'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.name = args['name']
        self.resume = args['resume']
        self.load_model = args['load_model']
        self.test_only = args['train']['test_only']
        self.cpu = args['sys']['cpu']

        # Log directory
        if self.test_only:
            assert 'load' in args and args['load'] is not None, \
                'Test_only mode requires at least one loading directory (load={name})'
            load_name = args['load']
            self.log_dir = os.path.join(args['home_dir'], 'runs', load_name, self.name)
            assert os.path.exists(os.path.join(args['home_dir'], 'runs', load_name)), \
                f'Cant find {load_name}.'
        else: 
            load_name = args['load'] if 'load' in args else None     
            self.log_dir = os.path.join(args['home_dir'], 'runs', self.name)

        if type(load_name) != list:
            load_name = [load_name]
            
        self.load_dirs = [os.path.join(args['home_dir'], 'runs', ln) if ln is not None else self.log_dir\
                             for ln in load_name]

        # Delete the runs/{args.name} directory
        if 'reset' in args and args['reset']:
            os.system('rm -rf ' + self.log_dir)

        # Creating directories
        os.makedirs(self.log_dir, exist_ok=True)
        if args['train']['save_results'] or args['train']['save_kernels']:
            self.img_dir = os.path.join(self.log_dir, 'results')
            os.makedirs(self.img_dir, exist_ok=True)

        # Initializing logging
        handlers = []
        handlers.append(logging.FileHandler(filename=f'{self.log_dir}/run.log'))
        handlers.append(logging.StreamHandler(sys.stdout)) 
        
        level = logging.DEBUG if args['debug'] else logging.INFO
        
        logging.basicConfig(
            level=level,
            handlers=handlers)

        if args['debug']:
            logging.debug(f'[*] DEBUG MODE ON')
        logging.info(f'[*] Checkpoint directory: {self.log_dir}.')
        logging.info(f'[*] Loading directories: {self.load_dirs}.')
        # Saving yaml config if it doesn't exist.
        if not os.path.exists(os.path.join(self.log_dir, 'config.yaml')):
            with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
                logger.info('[*] Saving run\'s yaml config. Load/reset/resume/run is not saved if specified')
                args_without_load = copy.deepcopy(args)
                dont_save = ['load', 'reset', 'resume', 'debug', 'run']
                for a in dont_save:
                    args_without_load.pop(a, None)
                yaml.dump(args_without_load, f)

    def load_checkpoint(self, models, optimizers=None, schedulers=None):
        for load_dir in self.load_dirs:
            assert os.path.exists(load_dir), f'{load_dir} not found.'
            torch_path = os.path.join(load_dir, f'{str(self.load_model)}.pt')

            if not os.path.exists(torch_path):
                logger.info(f'[*] {self.load_model}.pt model not found. Trying to load latest model instead.')
                torch_path = os.path.join(load_dir, f'latest.pt')

            step = 0
            loss = []
            if os.path.exists(torch_path):
                logger.info(f'[*] Loading from {torch_path}. ')
                checkpoint = torch.load(torch_path) if not self.cpu else torch.load(torch_path, map_location=torch.device('cpu'))
                for model_typ in models.keys():
                    if model_typ in checkpoint['models']:
                        self._load_model(models[model_typ], checkpoint['models'][model_typ])  
                        logger.info(f'[*] {model_typ} model loaded.')
                    if 'models_opt' in checkpoint and model_typ in checkpoint['models_opt']:
                        if not hasattr(models[model_typ], 'adapt_opt'):
                            logger.warn(f'[********] INNER-LOOP OPTIMIZER FOUND IN {load_dir} CANT BE LOADED.')
                        else:
                            models[model_typ].adapt_opt.load_state_dict(checkpoint['models_opt'][model_typ]) 
                            logger.info(f'[*] {model_typ} model inner-loop optimizer state dict loaded.')
                if optimizers is not None and self.resume:
                    for opt_typ in optimizers.keys():
                        if opt_typ in checkpoint['optimizers']:
                            optimizers[opt_typ].load_state_dict(checkpoint['optimizers'][opt_typ])
                            logger.info(f'[*] {opt_typ} optimizer loaded.')
                    step = checkpoint['step']
                    loss = checkpoint['loss']
                elif optimizers is not None and not self.test_only:
                    logger.warn(f'[********] RESUME FLAG IS NOT SET. IGNORE THIS IF TRAINING FROM SCRATCH.') 
                if schedulers is not None and self.resume:
                    for sch_typ in schedulers.keys():
                        if sch_typ in checkpoint['schedulers']:
                            schedulers[sch_typ].load_state_dict(checkpoint['schedulers'][sch_typ])
                            logger.info(f'[*] {sch_typ} scheduler loaded.')
            else:
                logger.info(f'[*] No existing checkpoint saved. Training from scratch.')
        
        return step, loss
    
    def _load_model(self, model, state_dict):
        own_state = model.state_dict()
        logger.debug(f"[*] Own state keys: {own_state.keys()}")

        def _copy_state(k, param):
            nonlocal own_state
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            if own_state[k].size() != param.size():
                logger.info(f'[*] {k} is found but the parameter size is different. skipping..')
            else:
                logger.info(f'[*] {k} loaded successfully')
                own_state[k].copy_(param)

        for k, v in state_dict.items():
            logger.debug(f'Trying to load {k}')
            ok = k
            if k.startswith('model.'):
                logger.debug(f'[*] Old model key detected: {k}. Running backward compatibility')
                k = k.split('.', 1)[1]

            if k.startswith('module.model.'):
                logger.debug(f'[*] Old model key detected: {k}. Running backward compatibility')
                tmp = k.split('.', 2)
                k = tmp[0] + '.' + tmp[2]

            if k in own_state:
                _copy_state(k, v)
            else:
                if hasattr(model, 'load_from_base_model'):
                    logger.debug(f'[*] Attempt 1: Run base model compatibility')
                    k = model.load_from_base_model(k)
                    if k in own_state:
                        _copy_state(k, v)
                        continue
                
                if hasattr(model, 'backward_compatibility'):
                    logger.debug(f'[*] Attempt 2: Run backward model compatibility')
                    k = model.backward_compatibility(k)
                    if k in own_state:
                        _copy_state(k, v)
                        continue

                logger.debug(f'[*] Attempt 3: Removing module.')
                if k.startswith('module.'):                    
                    k = k[7:]
                    if k in own_state:
                        _copy_state(k, v)
                        continue

                logger.debug(f'[*] Attempt 4: Appending module.')
                k = f'module.{k}'
                if k in own_state:
                    _copy_state(k, v)
                    continue
                else:
                    logger.error(f'[*] {ok} not found. Please double-check. You might be using a really old model')
        
        model.load_state_dict(own_state)

    
    def save_checkpoint(self, step, loss, models, optimizers, schedulers=None):
        
        def _save_checkpoint(path, step, loss, models, optimizers, schedulers):
            save_params = {
                'step': step,
                'loss': loss,
                'models': {k: v.state_dict() for k, v in models.items()},
                'models_opt': {},
                'optimizers': {k: v.state_dict() for k, v in optimizers.items()}
            }
            if schedulers is not None:
                save_params['schedulers'] = {k: v.state_dict() for k, v in schedulers.items()}
            
            for model_typ, model in models.items():
                if hasattr(model, 'adapt_opt'):
                    save_params['models_opt'][model_typ] = model.adapt_opt.state_dict()
            torch.save(save_params, path)

        logger.info(f'[*] Saving latest model at step {step} to {self.log_dir}.')
        _save_checkpoint(os.path.join(self.log_dir, f'{step}.pt'), step, loss, models, optimizers, schedulers)
        _save_checkpoint(os.path.join(self.log_dir, f'latest.pt'), step, loss, models, optimizers, schedulers)

    def write_step(self, step):
        with open(os.path.join(self.log_dir, 'step.txt'), 'w') as f:
            f.write(str(step))

    def save_kernel(self, kernel, fn):
        sio.savemat(os.path.join(self.img_dir, fn), {'kernel': kernel})

    def save_results(self, img, fn):
        assert len(img.size()) == 4 and img.size(0) == 1
        img = img[0].permute(1, 2, 0).cpu().numpy()

        imageio.imwrite(os.path.join(self.img_dir, fn), img)

    def dump_results(self, quant_result):
        logger.info(f'[*] Quantitative results are saved to {os.path.join(self.log_dir, "quant_result.pickle")}')
        with open(os.path.join(self.log_dir, "quant_result.pickle"), 'wb') as f:
            pickle.dump(quant_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_intermediate_results(self, img, fn):
        int_res_dir = os.path.join(self.log_dir, 'intermediate_results')
        os.makedirs(int_res_dir, exist_ok=True)
        img = img[0].permute(1, 2, 0).cpu().numpy()

        imageio.imwrite(os.path.join(int_res_dir, fn), img)

    def dump_crash(self, model, quit=True):
        logger.error(f'[*] Run failed as model diverged. Diverged model is saved to {os.path.join(self.log_dir, "crash.pt")}')
        torch.save(model.state_dict(), os.path.join(self.log_dir, "crash.pt"))
        if quit:
            sys.exit()
