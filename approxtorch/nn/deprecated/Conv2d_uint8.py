import torch
from . import conv2d_uint8
from torch.nn.modules.utils import _pair

class Conv2d_uint8(torch.nn.Module):
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                kernel_size: int | tuple[int, int], 
                lut: torch.Tensor,
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                scale_feature: torch.Tensor | None = None,
                zero_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                zero_weight: torch.Tensor | None = None,
                grad: str = 'ste',
                grad_data = None,
                bias: bool | torch.Tensor = True,
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.grad = grad
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.qmethod = qmethod
        self.bias = bias
        self.has_bias = bias is not None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.frozen_scale = True
        match qmethod[0]:
            case 'static':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=False)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=False)
                self.zero_feature = torch.nn.Parameter(zero_feature, requires_grad=False)
                self.zero_weight = torch.nn.Parameter(zero_weight, requires_grad=False)
            case 'dynamic':
                self.scale_feature = None
                self.scale_weight = None
            case 'trainable':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=True)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=True)
            case _:
                raise ValueError("Invalid quantization method")
        
        match grad:
            case "ste":
                self.grad_data = None
            case 'lre' | 'custom':
                self.grad_lut_dx = torch.nn.Parameter(grad_data[0], requires_grad=False)
                self.grad_lut_dy = torch.nn.Parameter(grad_data[1], requires_grad=False)
                self.grad_data = (self.grad_lut_dx, self.grad_lut_dy)
            case 'half_custom':
                self.grad_lut_dy = torch.nn.Parameter(grad_data, requires_grad=False)
                self.grad_data = self.grad_lut_dy
            case _:
                raise ValueError("Invalid gradient method")
        
        if isinstance(self.bias, torch.Tensor):
            self.bias = torch.nn.Parameter(self.bias)
            self.has_bias = True
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.has_bias = True
        elif bias == False or bias == None:
            self.bias = None
            self.has_bias = False
        else:
            raise ValueError("Invalid bias type")
    
    def __repr__(self):
        return f"Conv2d_uint8(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
            f"kernel_size={self.kernel_size}, qmethod={self.qmethod}, grad={self.grad}, " \
            f"bias={self.has_bias}, stride={self.stride}, padding={self.padding}, " \
            f"dilation={self.dilation}, groups={self.groups}, freeze_scales={self.frozen_scale})"
                
    def updata_scale(self, x, weight, qmethod):
        max_feature = torch.max(x)
        min_feature = torch.min(x)
        new_scale_feature = (max_feature - min_feature) / 255.
        new_zero_feature = - torch.round(min_feature / new_scale_feature)
        if qmethod[2] == 'channel':
            max_weight = torch.amax(weight, dim=(1,2,3), keepdim=False)
            min_weight = torch.amin(weight, dim=(1,2,3), keepdim=False)
            new_scale_weight = (max_weight - min_weight) / 255.
            new_zero_weight = - torch.round(min_weight / new_scale_weight)
        elif qmethod[2] == 'tensor':
            max_weight = torch.max(weight)
            min_weight = torch.min(weight)
            new_scale_weight = (max_weight - min_weight) / 255.
            new_zero_weight = - torch.round(min_weight / new_scale_weight)
        
        new_scale_feature = 0.95 * self.scale_feature + 0.05 * new_scale_feature
        new_zero_feature = 0.95 * self.zero_feature + 0.05 * new_zero_feature
        new_scale_weight = 0.95 * self.scale_weight + 0.05 * new_scale_weight
        new_zero_weight = 0.95 * self.zero_weight + 0.05 * new_zero_weight
        
        with torch.no_grad():
            self.scale_feature.copy_(new_scale_feature)
            self.scale_weight.copy_(new_scale_weight)
            self.zero_feature.copy_(new_zero_feature)
            self.zero_weight.copy_(new_zero_weight)

    def freeze_scale(self):
        self.frozen_scale = True
    
    def unfreeze_scale(self):
        self.frozen_scale = False
        
        
    def forward(self, x):
        if not self.frozen_scale and self.qmethod[0] == 'static':
            self.updata_scale(x, self.weight, self.qmethod)
        return  conv2d_uint8(x, 
                        self.weight,
                        self.lut,
                        self.qmethod,
                        self.scale_feature,
                        self.zero_feature,
                        self.scale_weight,
                        self.zero_weight,
                        self.grad,
                        self.grad_data,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)