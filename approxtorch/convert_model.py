import torch
import torch.nn as nn
from approxtorch.nn import Conv2d_int8, Conv2d_uint8
# from approxtorch.new_nn import Conv2d_uint8_STE, Conv2d_uint8_custom 
from typing import Literal

# this function convert the model into approximate model

def convert_model(model, 
                lut,
                qtype: str = 'int8',
                x_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
                w_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
                grad: str = 'ste',
                grad_dx: torch.Tensor | None = None,
                grad_dy: torch.Tensor | None = None,
                conv_only = True,
                ignore_first_conv = True
            ):
    
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
                        
            new_module = None
            # check if this model is Normal Conv2d or Depthwise_Conv2d
            match qtype:
                case 'uint8':
                    new_module = Conv2d_uint8(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        lut = lut,
                        x_quantizer = x_quantizer,
                        w_quantizer = w_quantizer,
                        update_qparams = False,
                        eps = 0.05,
                        grad = grad,
                        grad_dx = grad_dx,
                        grad_dy = grad_dy,
                        bias = bias,
                        stride = stride,
                        padding = padding,
                        dilation = dilation,
                        groups = groups
                    )
                case 'int8':
                    pass
            modules_to_replace.append((name, new_module))
 
        
    for name, new_module in modules_to_replace:
        parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent_module = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent_module, attr_name, new_module)
        
    return model

