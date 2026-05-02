import torch
from . import quantizer as Q
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair

class Conv2d_int8(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 x_quantizer:str = 'symmetric',
                 w_quantizer:str = 'asymmetric',
                 grad: str = 'ste',
                 grad_dx: torch.Tensor | None = None,
                 grad_dy: torch.Tensor | None = None,
                 bias: torch.Tensor | None = None,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 update_scale: bool = True,
                 scale_momentum: float = 0.05
         ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.x_quantizer = x_quantizer
        self.w_quantizer = w_quantizer
        self.grad = grad
        self.qmin = -127
        self.qmax = 127
        self.scale_momentum = scale_momentum
        self.update_scale = update_scale  # whether to update scale during training, used for BatchNorm fusion
        
        # lut 
        self.register_buffer('lut', lut)
        # weight
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        # quantization parameters
        
        match x_quantizer:
            case 'symmetric':
                self.register_buffer('scale_x', torch.tensor(1.0))
                self.zero_x = None  # 占个位置 没用
            case 'asymmetric':
                raise NotImplementedError("asymmetric quantization for x is not implemented yet")
            case _:
                raise ValueError("Invalid quantization method for x")
        
        self.register_buffer('scale_w', torch.tensor(1.0))
        self.zero_w = None  # 占个位置 没用

        # bias
        if isinstance(bias, torch.Tensor):
            self.bias = nn.Parameter(bias)
        elif bias == True:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")

        if self.grad == 'custom' or self.grad == 'lre':
            self.register_buffer('grad_dx', grad_dx)
            self.register_buffer('grad_dy', grad_dy)
        else:
            self.grad_dx = None
            self.grad_dy = None

    def __repr__(self):
        return f"Conv2d_int8(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"x_quantizer={self.x_quantizer}, w_quantizer={self.w_quantizer}, grad={self.grad})"
    

    def unfreeze_scale(self):
        self.update_scale = True
    def freeze_scale(self):
        self.update_scale = False

    def _update_scale(self, x):
        with torch.no_grad():
            abs_max = x.abs().max()
            current_scale = abs_max / ((self.qmax - self.qmin) / 2 ) 
            new_scale = self.scale_momentum * current_scale + (1 - self.scale_momentum) * self.scale_x
            self.scale_x.copy_(new_scale)



    def forward(self, x: torch.Tensor):
        
        if self.update_scale and self.training:
            self._update_scale(x)

        output = conv2d_int8(x, self.weight, self.lut, self.grad, self.grad_dx, self.grad_dy, 
                            self.x_quantizer, self.w_quantizer, 
                            self.scale_x, self.zero_x, self.scale_w, self.zero_w, self.bias,
                            self.stride, self.padding, self.dilation, self.groups, self.qmin, self.qmax)
        
        return output