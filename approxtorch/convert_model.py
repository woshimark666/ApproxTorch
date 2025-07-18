import torch
import torch.nn as nn
from approxtorch.nn import Conv2d_int8_STE, Linear_int8_STE, Conv2d_uint8_STE, Linear_uint8_STE
from typing import Literal

# this function convert the model into approximate model
# 暂时我只写了转换为dynamic quant的转换公式，static 还要考虑一下
def convert_model(model, 
                  lut,
                  qtype: Literal['int8', 'uint8'] = 'int8',
                  qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'tensor'),
                  gradient_lut=None, 
                  gradient='ste',
                  conv_only=True,
                  ignore_first_conv=True
                  ):
    if gradient.lower() not in ['ste', 'est']:
        raise ValueError("gradient parameter must be either 'ste' or 'est'")
    
    modules_to_replace = []
    conv2d_count = 0  # 新增：用于计数Conv2d层
    # print(qtype, qmethod)
    if qtype == 'int8':
        Conv2d = Conv2d_int8_STE
        Linear = Linear_int8_STE
    elif qtype == 'uint8':
        Conv2d = Conv2d_uint8_STE
        Linear = Linear_uint8_STE
    
    if gradient.lower() == 'ste':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv2d_count += 1
                if ignore_first_conv and conv2d_count == 1:
                    continue  # 跳过第一个Conv2d层
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                bias = module.bias
                new_module = Conv2d(in_channels, 
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
                
            elif isinstance(module, nn.Linear) and not conv_only:
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias
                new_module = Linear(in_features,
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
