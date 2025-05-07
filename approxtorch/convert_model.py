import torch
import torch.nn as nn
from approxtorch.layer import Conv2d_int8, Conv2d_int8_est, Linear_int8, Linear_int8_est 

# this function convert the model into int8 approximate model
def convert_model(model, conv2d_module, linear_module, lut, gradient_lut=None, gradient='ste'):
    if gradient.lower() not in ['ste', 'est']:
        raise ValueError("gradient parameter must be either 'ste' or 'est'")
    
    modules_to_replace = []

    if gradient.lower() == 'ste':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0]
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                bias = module.bias
                new_module = Conv2d_int8(in_channels, out_channels, kernel_size, 
                                                lut, bias, stride, padding, dilation)
                modules_to_replace.append((name, new_module))
                
            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias
                new_module = Linear_int8(in_features, out_features, lut, bias)
                modules_to_replace.append((name, new_module))
    
    elif gradient.lower() == 'est':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0]
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                bias = module.bias
                new_module = Conv2d_int8_est(in_channels, out_channels, kernel_size, 
                                                lut, gradient_lut, bias, stride, padding, dilation)
                modules_to_replace.append((name, new_module))
                
            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias
                new_module = Linear_int8_est(in_features, out_features, lut, gradient_lut, bias)
                modules_to_replace.append((name, new_module))
    
    
    for name, new_module in modules_to_replace:
        parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent_module = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent_module, attr_name, new_module)
    return model