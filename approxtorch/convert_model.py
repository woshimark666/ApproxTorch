import torch
import torch.nn as nn
from approxtorch.nn import Conv2d_int8_STE, Linear_int8_STE
from typing import Literal

# this function convert the model into approximate model
def convert_model(model, 
                  lut,
                  qtype: Literal['int8', 'uint8'] = 'int8',
                  qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'tensor'),
                  gradient_lut=None, 
                  gradient='ste'):
    if gradient.lower() not in ['ste', 'est']:
        raise ValueError("gradient parameter must be either 'ste' or 'est'")
    
    modules_to_replace = []
    
    if gradient.lower() == 'ste':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                bias = module.bias
                new_module = Conv2d_int8_STE(in_channels, 
                                out_channels,
                                kernel_size,
                                lut,
                                qmethod,
                                qparams=None,
                                bias=bias,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
                # Transfer weights
                new_module.weight.data.copy_(module.weight.data)
                if bias is not None:
                    new_module.bias.data.copy_(module.bias.data)
                modules_to_replace.append((name, new_module))
                
            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias
                new_module = Linear_int8_STE(in_features,
                                            out_features,
                                            lut,
                                            qmethod[0],
                                            qparams=None,
                                            bias=bias)
                # Transfer weights
                new_module.weight.data.copy_(module.weight.data)
                if bias is not None:
                    new_module.bias.data.copy_(module.bias.data)
                modules_to_replace.append((name, new_module))
    
        for name, new_module in modules_to_replace:
            parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent_module, attr_name, new_module)
    return model
