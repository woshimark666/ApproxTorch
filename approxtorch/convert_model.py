import torch
import torch.nn as nn
# 旧实现（Conv2d_uint8 / 旧 Conv2d_int8 / gradual）已移入
# approxtorch/nn/deprecated/，不再可从包导入；本文件只保留 decoupled 路径。
from approxtorch.nn import Conv2d_int8

# this function convert the model into approximate model

def to_qat_int8(
        model: nn.Module,
        lut: torch.Tensor,
        x_quantizer: str = 'symmetric',
        w_quantizer: str = 'symmetric',
        grad: str = 'ste',
        dx: torch.Tensor | None = None,
        dw: torch.Tensor | None = None,
        conv_only: bool = True,
        ignore_first_conv: bool = True,
        scale_momentum: float = 0.05,
        decoupled: bool = True,
        weight_bits: int = 8
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



            if decoupled:
                new_module = Conv2d_int8(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    lut=lut,
                    x_quantizer=x_quantizer,
                    w_quantizer=w_quantizer,
                    grad=grad,
                    dx=dx,
                    dw=dw,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    update_scale=True,
                    scale_momentum=scale_momentum,
                    weight_bits=weight_bits,
                )
                with torch.no_grad():
                    new_module.weight.copy_(module.weight)
                    new_module._reset_scale_w_from_weight()
            else:
                raise NotImplementedError(
                    "非 decoupled 的旧 Conv2d_int8 已移入 approxtorch/nn/deprecated/"
                    "（Conv2d_int8_v2.py）；请改用 to_qat_int8(..., decoupled=True)")
            
            modules_to_replace.append((name, new_module))

        
    for name, new_module in modules_to_replace:
        parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent_module = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent_module, attr_name, new_module)
        
    return model



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
    # 旧 API：依赖的 Conv2d_uint8 / 旧 Conv2d_int8 已移入 approxtorch/nn/deprecated/。
    # 保留函数签名只为不破坏 `from .convert_model import convert_model` 的包导入。
    raise NotImplementedError(
        "convert_model 的旧实现已废弃（Conv2d_uint8 / 旧 Conv2d_int8 移入 "
        "approxtorch/nn/deprecated/）；请改用 to_qat_int8(..., decoupled=True)")
