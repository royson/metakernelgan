import traceback
import torch
import copy
from torch.autograd import grad
from trainer import kernelgan
from learn2learn.algorithms.maml import MAML
from learn2learn.utils import clone_module, update_module

import logging
logger = logging.getLogger(__name__)

def maml_update(model, lr, grads=None):
    if grads is not None:        
        params = list(model.parameters())
        if not len(grads) == len(params):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)

class MAMLWrapper(MAML):
    def __init__(self, model, lr, *args, gradient_clip=None, \
                    adapt_opt_sd=None, adapt_opt_args=None, **kwargs):
        super(MAMLWrapper, self).__init__(model, lr, *args, **kwargs)
        self.gradient_clip = gradient_clip

        self.adapt_opt_sd = adapt_opt_sd
        self.adapt_opt_args = adapt_opt_args
        if self.adapt_opt_sd is not None:
            self.adapt_opt = torch.optim.Adam(self.module.parameters(), lr=lr, **adapt_opt_args) 
            self.adapt_opt.load_state_dict(self.adapt_opt_sd)        

    def set_lr(self, lr): 
        if self.adapt_opt_sd is not None:
            for g in self.adapt_opt.param_groups:
                g['lr'] = lr
        else:
            self.lr = lr

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        
        if self.adapt_opt_sd is not None:
            model = copy.deepcopy(self.module)
            if model.model_name.lower() == 'discriminator':
                model.apply(kernelgan.weights_init_D)
        else:
            model = clone_module(self.module) 
        return MAMLWrapper(model,
                    lr=self.lr,
                    gradient_clip=self.gradient_clip,
                    adapt_opt_sd=self.adapt_opt_sd,
                    adapt_opt_args=self.adapt_opt_args,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)

    def adapt(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None,
              retain_graph=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order
        if retain_graph is None:
            retain_graph = True if second_order else False

        # Update the module
        if self.adapt_opt_sd is not None:
            assert first_order
            self.adapt_opt.zero_grad()
            loss.backward(retain_graph=retain_graph)
            self.adapt_opt.step()
            self.adapt_opt.zero_grad() # remove gradients from last inner loop step for outer loop update
        else: 
            model_params = list(self.module.parameters())

            if allow_nograd:
                # Compute relevant gradients
                diff_params = [p for p in model_params if p.requires_grad]
                grad_params = grad(loss,
                                diff_params,
                                retain_graph=second_order or retain_graph,
                                create_graph=second_order,
                                allow_unused=allow_unused)
                gradients = []
                grad_counter = 0

                # Handles gradients for non-differentiable parameters
                for param in model_params:
                    if param.requires_grad:
                        gradient = grad_params[grad_counter]
                        grad_counter += 1
                    else:
                        gradient = None
                    gradients.append(gradient)
            else:
                try:
                    gradients = grad(loss,
                                    model_params,
                                    retain_graph=second_order or retain_graph,
                                    create_graph=second_order,
                                    allow_unused=allow_unused)
                except RuntimeError:
                    traceback.print_exc()
                    print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

            if self.gradient_clip is not None and self.gradient_clip > 0:
                gradients = [torch.clamp(g, -self.gradient_clip, self.gradient_clip) \
                                if g is not None else None for g in gradients]
            
            self.module = maml_update(self.module, self.lr, grads=gradients)
