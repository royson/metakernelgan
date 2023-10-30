import torch
import model.utils as mutils
from utils import AttrDict
from pprint import pformat

import logging
logger = logging.getLogger(__name__)

def _get_model(args, model_name, model_args, optim_args):
    model_name = model_name.upper()
    if model_args is None:
        model_args = {}

    if model_name == 'DOWNSAMPLING':
        from model.downsampling_model import DownsamplingModel
        m = DownsamplingModel
    elif model_name == 'DISCRIMINATOR':
        from model.discriminator import Discriminator
        m = Discriminator
    else:
        raise NotImplementedError()

    device = torch.device('cpu' if args.sys.cpu else 'cuda')
    model = m(args, **model_args).to(device)
    optim_args = {**mutils.get_model_optim_args_default_values(), **optim_args}
    optim_args = AttrDict(optim_args)
    model.optim_args = optim_args
    model.model_name = model_name
    logger.info(f'{model_name} created. No. of parameters: {mutils.no_of_params(model)}. Optimization Args: {pformat(optim_args)}.')
    logger.debug(model)
    
    return model

def get_models(args):
    '''
        Models are grouped into two components: downsampling (generator), and discriminator
        All model related parameters are stored in the model itself: namely, the LR, LR decay, and steps
        
        Returns a dict of models with the associated keys
    '''
    model_components = ['downsampling', 'discriminator']
    models = {}
    for mc in model_components:
        if hasattr(args.model, mc):
            md = getattr(args.model, mc)
            models[mc] = _get_model(args, md.model_name, md.model_args, md.optim_args)    

    return models