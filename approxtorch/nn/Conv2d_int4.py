import torch
from . import conv2d_int4
from torch.nn.modules.utils import _pair

# exact class for int4
class Conv2d_int4_exact(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                 scale_feature: torch.Tensor | None = None,
                 scale_weight: torch.Tensor | None = None,
                 bias: bool | torch.Tensor = True,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1):
        
        super().__init__()
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
            case 'dynamic':
                self.scale_feature = None
                self.scale_weight = None
            case 'trainable':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=True)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=True)
            case _:
                raise ValueError("Invalid quantization method")
        
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
        return f"Conv2d_int4_exact(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, qmethod={self.qmethod}, " \
               f"bias={self.has_bias}, stride={self.stride}, padding={self.padding}, " \
               f"dilation={self.dilation}, groups={self.groups}, freeze_scales={self.frozen_scale})"
    
    
    def updata_scale(self, x, weight):
        absmax_feature = torch.abs(x).max()
        absmax_weight = torch.abs(weight).max()
        new_scale_feature = 0.95 * self.scale_feature + 0.05 * (absmax_feature/7.5)
        new_scale_weight = 0.95 * self.scale_weight + 0.05 * (absmax_weight/7.5)

        with torch.no_grad():
            self.scale_feature.copy_(new_scale_feature)
            self.scale_weight.copy_(new_scale_weight)
    
    def freeze_scale(self):
        self.frozen_scale = True
    
    def unfreeze_scale(self):
        self.frozen_scale = False
    
    def forward(self, x):
        if not self.frozen_scale and self.qmethod[0] == 'static':
            self.updata_scale(x, self.weight)
            
        return conv2d_int4.conv2d_int4_exact(x,
                                          self.weight,
                                          self.qmethod,
                                          self.scale_feature,
                                          self.scale_weight,
                                          self.bias, self.stride, self.padding, self.dilation, self.groups)
        
# class for approximate int4 with STE gradient
class Conv2d_int4_STE(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                 scale_feature: torch.Tensor | None = None,
                 scale_weight: torch.Tensor | None = None,
                 bias: bool | torch.Tensor = True,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1):
        
        super().__init__()
        self.register_buffer('lut', lut)
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
            case 'dynamic':
                self.scale_feature = None
                self.scale_weight = None
            case 'trainable':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=True)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=True)
            case _:
                raise ValueError("Invalid quantization method")
        
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
        return f"Conv2d_int4_STE(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, qmethod={self.qmethod}, " \
               f"bias={self.has_bias}, stride={self.stride}, padding={self.padding}, " \
               f"dilation={self.dilation}, groups={self.groups}, freeze_scales={self.frozen_scale})"
    
    
    def updata_scale(self, x, weight):
        absmax_feature = torch.abs(x).max()
        absmax_weight = torch.abs(weight).max()
        new_scale_feature = 0.95 * self.scale_feature + 0.05 * (absmax_feature/7.5)
        new_scale_weight = 0.95 * self.scale_weight + 0.05 * (absmax_weight/7.5)

        with torch.no_grad():
            self.scale_feature.copy_(new_scale_feature)
            self.scale_weight.copy_(new_scale_weight)
    
    def freeze_scale(self):
        self.frozen_scale = True
    
    def unfreeze_scale(self):
        self.frozen_scale = False
    
    def forward(self, x):
        if not self.frozen_scale and self.qmethod[0] == 'static':
            self.updata_scale(x, self.weight)
            
        return conv2d_int4.conv2d_int4_STE(x,
                                          self.weight,
                                          self.lut,
                                          self.qmethod,
                                          self.scale_feature,
                                          self.scale_weight,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups)
        
        
class Conv2d_int4_BIT(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 gradient_lut: tuple[torch.Tensor, torch.Tensor],
                 qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                 scale_feature: torch.Tensor | None = None,
                 scale_weight: torch.Tensor | None = None,
                 bias: bool | torch.Tensor = True,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('gradient_lut_dx', gradient_lut[0])
        self.register_buffer('gradient_lut_dy', gradient_lut[1])
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
                # self.register_buffer('scale_feature', torch.randn(()))
                # self.register_buffer('scale_weight', torch.randn((out_channels)))
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=False)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=False)
            case 'dynamic':
                self.scale_feature = None
                self.scale_weight = None
            case 'trainable':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=True)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=True)
            case _:
                raise ValueError("Invalid quantization method")
        
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
        return f"Conv2d_int4_BIT(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, qmethod={self.qmethod}, " \
               f"bias={self.has_bias}, stride={self.stride}, padding={self.padding}, " \
               f"dilation={self.dilation}, groups={self.groups}, freeze_scales={self.frozen_scale})"
    
    
    def updata_scale(self, x, weight):
        absmax_feature = torch.abs(x).max()
        absmax_weight = torch.abs(weight).max()
        old_absmax_feature = self.scale_feature * 7.5
        old_absmax_weight = self.scale_weight * 7.5
        new_absmax_feature = 0.95 * old_absmax_feature + 0.05 * absmax_feature
        new_absmax_weight = 0.95 * old_absmax_weight + 0.05 * absmax_weight
        new_scale_feature = new_absmax_feature / 7.5
        new_scale_weight = new_absmax_weight / 7.5

        with torch.no_grad():
            self.scale_feature.copy_(new_scale_feature)
            self.scale_weight.copy_(new_scale_weight)
    
    def freeze_scale(self):
        self.frozen_scale = True
    
    def unfreeze_scale(self):
        self.frozen_scale = False
    
    def forward(self, x):
        if not self.frozen_scale and self.qmethod[0] == 'static':
            self.updata_scale(x, self.weight)
            
        return conv2d_int4.conv2d_int4_bit(x,
                                          self.weight,
                                          self.lut,
                                          (self.gradient_lut_dx, self.gradient_lut_dy),
                                          self.qmethod,
                                          self.scale_feature,
                                          self.scale_weight,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups)