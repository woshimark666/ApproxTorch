import torch
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import quantizer as Q
import math
from torch.autograd import Function

class _conv2d_fakeint8_base(Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                weight: torch.Tensor, 
                lut: torch.Tensor, 
                s_x: torch.Tensor | None, 
                coefficient: torch.Tensor | None,
                bias, 
                stride, 
                padding, 
                dilation, 
                groups, 
                qmin, 
                qmax,
        ):
        
        B, C, H, W = x.shape
        O, C, kH, kW = weight.shape
        kernel_size = (kH, kW)
        sH, sW = stride
        pH, pW = padding
        dH, dW = dilation
        OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        
        ctx.has_bias = bias is not None
        ctx.kernel_size = kernel_size
        ctx.output_shape = (B, O, OH, OW)
        ctx.input_shape = x.shape
        ctx.stride = (sH, sW)
        ctx.padding = (pH, pW)
        ctx.dilation = (dH, dW)
        ctx.groups = groups
        ctx.qmin = qmin
        ctx.qmax = qmax
        
        # 1. quantization and 2. im2col 
        # static quant for x
        x = x / s_x
        x = torch.clamp(x, qmin, qmax)
        # dynamic quant for weight
        s_w = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True) / qmax
        s_w = torch.clamp(s_w, min=1e-12)
        weight = weight / s_w
        weight = torch.clamp(weight, qmin, qmax)
        # im2col
        x = torch.nn.functional.unfold(x, kernel_size=kernel_size, dilation=(dH, dW), padding=(pH, pW), stride=(sH, sW)) # (N, CKK, L)
        weight = weight.view(O, -1)  # (O, CKK)
        # save for backward
        ctx.save_for_backward(x, weight)
        ctx.scale_x = s_x # (1)
        ctx.scale_w = s_w # (O, 1, 1, 1)
        ctx.coefficient = coefficient
        # round, finish quantization
        x = torch.round(x)
        weight = torch.round(weight)
        
        # 3. bgemm
        output = at.backend.ops.bgemm_fake_int8_gpt(x, weight, lut) # (N, O, L)
        output = output.view(B, O, OH, OW)
        
        # 4. de-quantization
        output = output * s_x * s_w.view(1, -1, 1, 1) # (N, O, OH, OW)
            
        # 5. add bias
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)
        
        return output



class _conv2d_int8_bqsg64_float(_conv2d_fakeint8_base):
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_x, grad_weight, grad_bias = None, None, None
        s_x = ctx.scale_x
        s_w = ctx.scale_w
        s_w = s_w.view(-1) # (O)
        coefficient = ctx.coefficient
        x, weight = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        (N, O, OH, OW) = ctx.output_shape
        (_, C, H, W) = ctx.input_shape
        stride = ctx.stride 
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        # x [N, CKK, L]
        # weight [O, CKK]
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
        grad_x, grad_weight = at.backend.ops.bgemm_bqsg64_float_backward(upstream_grad, x, weight, coefficient, s_x, s_w)
        # grad_x [N, CKK, L]
        # grad_weight [O, CKK]
        # fold back to original shape
        grad_x = torch.nn.functional.fold(grad_x, output_size=(H, W), kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride) # (N, C, H, W)
        grad_weight = grad_weight.view(O, C, kernel_size[0], kernel_size[1]) # (O, C, kH, kW)
        

        
        
        return grad_x, grad_weight, None, None, None, grad_bias, None, None, None, None, None, None

def conv2d_int8_bqsg64_float(x, weight, lut, s_x, coefficient, bias, stride, padding, dilation, groups, qmin, qmax):
    return _conv2d_int8_bqsg64_float.apply(x, weight, lut, s_x, coefficient, bias, stride, padding, dilation, groups, qmin, qmax)

class Conv2d_int8_BQSG64_float(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 coeffcient: torch.Tensor | None = None,
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
        self.qmin = -127
        self.qmax = 127
        self.scale_momentum = scale_momentum
        self.update_scale = update_scale  # whether to update scale during training, used for BatchNorm fusion
        
        # lut 
        self.register_buffer('lut', lut)
        # coefficient 
        self.register_buffer('coefficient', coeffcient)
        # scale_x 
        self.register_buffer('scale_x', torch.tensor(1.0))
        # weight
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        # bias
        if isinstance(bias, torch.Tensor):
            self.bias = nn.Parameter(bias)
        elif bias == True:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")

    def __repr__(self):
        return f"Conv2d_int8_BQSG64_float(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"update_scale={self.update_scale}, scale_momentum={self.scale_momentum})"
    
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

        output = conv2d_int8_bqsg64_float(x, self.weight, self.lut, self.scale_x, self.coefficient, self.bias, self.stride, self.padding, self.dilation, self.groups, self.qmin, self.qmax)

        return output