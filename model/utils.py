def get_model_optim_args_default_values():
    return {
        'meta_algorithm': 'maml',
        'meta_opt': 'adam',
        'meta_opt_args': None,
        'meta_lr': 0.0001,
        'meta_lr_decay': None,
        'task_opt': 'sgd',
        'task_opt_args': None,
        'first_order': True,
        'task_lr': 0.01,
        'task_lr_decay': None,
        'task_adapt_steps': 5,
        'task_gradient_clip': 0
        }
    
def no_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)