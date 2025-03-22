import torch
import torch.nn as nn


# this function convert the model into int8 approximate model
def convert_model(model, lut, conv2d_module, linear_module):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            bias = module.bias is not None
            new_module = conv2d_module(in_channels, out_channels, kernel_size, 
                                            lut, bias, stride, padding, dilation)
            modules_to_replace.append((name, new_module))
            
        elif isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            new_module = linear_module(in_features, out_features, lut, bias)
            modules_to_replace.append((name, new_module))
            
    for name, new_module in modules_to_replace:
        parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent_module = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent_module, attr_name, new_module)
    return model