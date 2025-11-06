import torch
import torch.nn as nn
from approxtorch.nn import Conv2d_int8
from typing import Literal

# this function convert the model into approximate model
def convert_model(model, 
                lut,
                qtype: str = 'int8',
                qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'tensor'),
                grad: str = 'ste',
                grad_data = None, 
                conv_only=True,
                ignore_first_conv=True
                ):
    
    
    # the first conv layer is ignored
    # the last linear layer is ignored
    # generally we do not quantize the first conv layer and the last linear layer
    
    
    modules_to_replace = []
    conv2d_count = 0 

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv2d_count += 1
            if ignore_first_conv and conv2d_count == 1:
                continue  # 跳过第一个Conv2d层
            
            # collect the Conv2d parameters
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            bias = module.bias
            groups = module.groups
            
            
            match qmethod:
                case ('static', 'tensor', 'tensor'):
                    scale_feature = torch.randn(())
                    scale_weight = torch.randn(())
                case ('static', 'tensor', 'channel'):
                    scale_feature = torch.randn(())
                    scale_weight = torch.randn(out_channels)
                case ('dynamic', _, _):
                    scale_feature = None
                    scale_weight = None
                case _:
                    raise ValueError("Invalid qmethod")
            
            new_module = None
            # check if this model is Normal Conv2d or Depthwise_Conv2d
            if qtype == 'int8':
                new_module = Conv2d_int8(
                    in_channels, out_channels, kernel_size, lut, qmethod, scale_feature, scale_weight, grad, grad_data, bias, stride, padding, dilation, groups)

            modules_to_replace.append((name, new_module))
        # we don't convert to Linear anymore.
                    
        # elif isinstance(module, nn.Linear) and not conv_only:
        #     in_features = module.in_features
        #     out_features = module.out_features
        #     bias = module.bias
        #     new_module = Linear(in_features,
        #                         out_features,
        #                         lut,
        #                         qmethod[0],
        #                         qparams=None,
        #                         bias=bias)
        #     # Transfer weights
        #     new_module.weight.data.copy_(module.weight.data)
        #     if bias is not None:
        #         new_module.bias.data.copy_(module.bias.data)
        #     modules_to_replace.append((name, new_module))

    for name, new_module in modules_to_replace:
        parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent_module = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent_module, attr_name, new_module)
        
    return model
