import torch
from . import fakequant
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import bgemm

class Conv2d_int8(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 x_quantizer:str = 'symmetric',
                 w_quantizer:str = 'symmetric',
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
        
        # delete the scale_w and zero_w since we will do dynamic quantization for weight

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
        return f"Conv2d_int8_decoupled(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
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

        # 0. compute output shape 
        B, C, H, W = x.shape
        O, C, kH, kW = self.weight.shape
        kernel_size = (kH, kW)
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        # 1. do quantization first 
        #   check if we need to update scale:
        if self.training and self.update_scale:
            self._update_scale(x)

        x = fakequant.symmetric_static_quantize_int8_per_tensor(x, self.scale_x, None, self.qmin, self.qmax)
        w, s_w = fakequant.symmetric_dynamic_quantize_int8_per_channel(self.weight, 0, self.qmin, self.qmax)
        
        # 2. im2col shape transform
        x = torch.nn.functional.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # (N, CKK, L)
        w = w.view(self.out_channels, -1) # (O, CKK)

        # 3. bgemm
        match self.grad:
            case 'ste':
                y = bgemm.bgemm_int8_ste(x, w, self.lut)
                # y [N, O, L]]
            case 'lre':
                raise NotImplementedError("lre gradient is not implemented yet")
            case 'custom':
                raise NotImplementedError("custom gradient is not implemented yet")
            case _:
                raise ValueError("Invalid gradient method")

        # 4. reshape and de-quantization
        y = y.view(B, O, OH, OW)
        y = y * self.scale_x.view(-1) * s_w.view(1, -1, 1, 1)

        return y