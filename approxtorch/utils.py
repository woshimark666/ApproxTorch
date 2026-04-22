import torch
import approxtorch as at

def set_alpha(module, alpha):
    
    for name, module in module.named_modules():
        if isinstance(module, at.nn.Conv2d_gradual_int8):
            module.update_alpha(alpha)

