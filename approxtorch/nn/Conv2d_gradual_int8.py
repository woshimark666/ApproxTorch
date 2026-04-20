import torch
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import quantizer as Q
import math
from torch.autograd import Function

class _conv2d_int8_gradual_base(Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                weight: torch.Tensor, 
                lut: torch.Tensor, 
                x_quantizer: str, 
                w_quantizer: str, 
                scale_x: torch.Tensor | None, 
                zero_x: torch.Tensor | None,
                alpha,  
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
        
        
        ctx.x_quantizer = x_quantizer
        ctx.w_quantizer = w_quantizer
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
        
        # 1. quantization
        if x_quantizer == 'symmetric':
            q_x = Q.symmetric_static_quantize_int8_per_tensor(x, scale_x, zero_x, qmin, qmax)
        elif x_quantizer == 'asymmetric':
            raise NotImplementedError("asymmetric quantization for x is not implemented yet")
        else:
            raise ValueError("quantization method for x is not supported")
            
            
        if w_quantizer == 'symmetric':
            q_w, scale_w, zero_w = Q.symmetric_dynamic_quantize_int8_per_channel(weight, qmin, qmax)
        else:
            raise ValueError("quantization method for weight is not supported")

        # 2. im2col
        q_x = at.backend.ops.im2col_int8(q_x, kernel_size, stride, padding, dilation)
        q_w = q_w.view(O, -1)
        
        
        # 3. bgemm
        output = at.backend.ops.bgemm_gradual_approx_int8(q_x, q_w, lut, alpha)

        # 4. de-quantization
        match (x_quantizer, w_quantizer):
            case ('symmetric', 'symmetric'):
                output = output * scale_x.view(-1) * scale_w.view(1, -1, 1)
            case ("asymmetric", 'symmetric'):
                raise NotImplementedError("asymmetric quantization for x is not implemented yet")
            case _:
                raise ValueError("Invalid quantization method")
            
        output = output.view(B, O, OH, OW)
        # 5. add bias
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)
        
        return output



class _conv2d_gradual_int8_ste(_conv2d_int8_gradual_base):
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_x, grad_weight, grad_bias = None, None, None
        x, weight = ctx.saved_tensors
        if ctx.has_bias and ctx.needs_input_grad[10]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
            
        grad_x = torch.nn.grad.conv2d_input(x.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        
        return grad_x, grad_weight, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None



class Conv2d_gradual_int8(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 x_quantizer:str = 'symmetric',
                 w_quantizer:str = 'symmetric',
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
        self.qmin = -127
        self.qmax = 127
        self.scale_momentum = scale_momentum
        self.update_scale = update_scale  # whether to update scale during training, used for BatchNorm fusion
        self.alpha = 0.0  # the alpha for gradual approximation, will be updated during training
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

    def __repr__(self):
        return f"Conv2d_int8_grad(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"x_quantizer={self.x_quantizer}, w_quantizer={self.w_quantizer}, update_scale={self.update_scale}, scale_momentum={self.scale_momentum}, alpha={self.alpha})"
    

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
        
        if self.update_scale:
            self._update_scale(x)

        output = _conv2d_gradual_int8_ste.apply(x, 
                                            self.weight, 
                                            self.lut, 
                                            self.x_quantizer, 
                                            self.w_quantizer, 
                                            self.scale_x, 
                                            self.zero_x, 
                                            self.alpha, 
                                            self.bias, 
                                            self.stride, 
                                            self.padding, 
                                            self.dilation, 
                                            self.groups, 
                                            self.qmin, 
                                            self.qmax)
        return output
    
    

