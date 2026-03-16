import torch
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import quantizer as Q
import math
from torch.autograd import Function

class _conv2d_int8_base(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, 
                lut: torch.Tensor, 
                grad: str, 
                dx_lut: torch.Tensor | None, 
                dw_lut: torch.Tensor | None,
                x_quantizer: tuple[str, str, str], 
                w_quantizer: tuple[str, str, str], 
                scale_x: torch.Tensor | None, 
                zero_x: torch.Tensor | None, 
                scale_w: torch.Tensor | None, 
                zero_w: torch.Tensor | None,
                bias, 
                stride, 
                padding, 
                dilation, 
                groups, 
                qmin, 
                qmax
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
        match x_quantizer:
            case ('dynamic', 'symmetric', 'tensor'):
                q_x, scale_x, zero_x = Q.symmetric_dynamic_quantize_int8_per_tensor(x, qmin, qmax)
            
            case _:
                raise ValueError("quantization method for x is not supported")
            
        match w_quantizer:
            case ('dynamic', 'symmetric', 'tensor'):
                q_w, scale_w, zero_w = Q.symmetric_dynamic_quantize_int8_per_tensor(weight, qmin, qmax)
            case ('dynamic', 'symmetric', 'channel'):
                q_w, scale_w, zero_w = Q.symmetric_dynamic_quantize_int8_per_channel(weight, qmin, qmax)
            case _:
                raise ValueError("quantization method for w is not supported")
        
        # 2. im2col
        q_x = at.backend.ops.im2col_int8(q_x, kernel_size, stride, padding, dilation)
        q_w = q_w.view(O, -1)
        
        match grad:
            case 'ste':
                ctx.save_for_backward(x, weight)
            case 'int_ste':
                ctx.save_for_backward(q_x, q_w)
                ctx.scale_x = scale_x
                ctx.zero_x = zero_x
                ctx.scale_w = scale_w
                ctx.zero_w = zero_w
            case 'custom':
                ctx.save_for_backward(q_x, q_w)
                ctx.dx_lut = dx_lut
                ctx.dw_lut = dw_lut
                ctx.scale_x = scale_x
                ctx.zero_x = zero_x
                ctx.scale_w = scale_w
                ctx.zero_w = zero_w
                
            case 'lre':
                ctx.save_for_backward(q_x, q_w)
                ctx.dx_lut = dx_lut
                ctx.dw_lut = dw_lut
                ctx.scale_x = scale_x
                ctx.zero_x = zero_x
                ctx.scale_w = scale_w
                ctx.zero_w = zero_w
            case _:
                raise ValueError("Invalid gradient type")
        
        # 3. bgemm
        output = at.backend.ops.bgemm_int8(q_x, q_w, lut)
        output = output.to(torch.float)

        # 4. de-quantization
        match (x_quantizer[1], w_quantizer[2]):
            case ('symmetric', 'tensor'):
                output = output * scale_x * scale_w
                
            case ('symmetric', 'channel'):
                # output (N, O, L), scale_x (), scale should be (1, O, 1)
                output = output * scale_x * scale_w.view(1, -1, 1)
                
            case _:
                raise ValueError("Invalid quantization method")
        output = output.view(B, O, OH, OW)
        # 5. add bias
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)
        
        return output



class _conv2d_int8_ste(_conv2d_int8_base):
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_x, grad_weight, grad_bias = None, None, None
        x, weight = ctx.saved_tensors
        if ctx.has_bias and ctx.needs_input_grad[12]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
            
        grad_x = torch.nn.grad.conv2d_input(x.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        return grad_x, grad_weight, None, None, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None


class _conv2d_int8_int_ste(_conv2d_int8_base):
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_x, grad_weight, grad_bias  = None, None, None
        q_x, q_w= ctx.saved_tensors
        q_x = q_x.to(torch.float)
        q_w = q_w.to(torch.float)
        scale_x, zero_x, scale_w, zero_w = ctx.scale_x, ctx.zero_x, ctx.scale_w, ctx.zero_w
    
        B, O, OH, OW = ctx.output_shape
        B, C, H, W = ctx.input_shape
        kH, kW = ctx.kernel_size
        L = OH * OW
        upstream_grad = upstream_grad.view(B,O,L)
        
        if ctx.has_bias and ctx.needs_input_grad[12]:
            grad_bias = upstream_grad.sum(dim=(0, 2))
            
        
        # upstream_grad (N, O, L)
        # q_x (N, CKK, L)
        # q_w (O, CKK)
        match (ctx.x_quantizer[1], ctx.w_quantizer[2]):
            case ('symmetric', 'tensor'):
                # grad_x 
                grad_x = torch.matmul(q_w.t(), upstream_grad * scale_w) # (N, CKK, L)
                grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation) # (N, C, H, W)
                
                # grad_weight
                grad_weight = torch.matmul(upstream_grad * scale_x, q_x.transpose(1, 2).contiguous()) # (N, O, CKK)
                grad_weight = grad_weight.sum(dim=0) # (O, CKK)
                grad_weight = grad_weight.view(O, C, kH, kW)
                
            case ('symmetric', 'channel'):
                grad_x = torch.matmul(q_w.t(), upstream_grad * scale_w.view(1, -1, 1)) # (N, CKK, L)
                grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation) # (N, C, H, W)
                
                # grad_weight
                grad_weight = torch.matmul(upstream_grad * scale_x, q_x.transpose(1, 2).contiguous()) # (N, O, CKK)
                grad_weight = grad_weight.sum(dim=0) # (O, CKK)
                grad_weight = grad_weight.view(O, C, kH, kW)
            case _:
                raise ValueError("Invalid quantization method")
        
        return grad_x, grad_weight, None, None, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None

class _conv2d_int8_lre(_conv2d_int8_base):
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_x, grad_weight, grad_bias  = None, None, None
        q_x, q_w= ctx.saved_tensors
        scale_x, zero_x, scale_w, zero_w = ctx.scale_x, ctx.zero_x, ctx.scale_w, ctx.zero_w
        dx_lut, dw_lut = ctx.dx_lut, ctx.dw_lut
        
        B, O, OH, OW = ctx.output_shape
        B, C, H, W = ctx.input_shape
        kH, kW = ctx.kernel_size
        L = OH * OW
        upstream_grad = upstream_grad.view(B,O,L)
        
        if ctx.has_bias and ctx.needs_input_grad[12]:
            grad_bias = upstream_grad.sum(dim=(0, 2))
            
        
        # upstream_grad (N, O, L)
        # q_x (N, CKK, L)
        # q_w (O, CKK)
        match (ctx.x_quantizer[1], ctx.w_quantizer[2]):
            case ('symmetric', 'tensor'):
                # grad_x 
                q_w = at.backend.ops.lut_lookup_int8(q_w, dx_lut)
                grad_x = torch.matmul(q_w.t().contiguous(), upstream_grad * scale_w) # (N, CKK, L)
                grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation) # (N, C, H, W)
                
                # grad_weight
                q_x = at.backend.ops.lut_lookup_int8(q_x, dw_lut)
                grad_weight = torch.matmul(upstream_grad * scale_x, q_x.transpose(1, 2).contiguous()) # (N, O, CKK)
                grad_weight = grad_weight.sum(dim=0) # (O, CKK)
                grad_weight = grad_weight.view(O, C, kH, kW)
                
            case ('symmetric', 'channel'):
                # grad_x
                q_w = at.backend.ops.lut_lookup_int8(q_w, dx_lut)
                grad_x = torch.matmul(q_w.t().contiguous(), upstream_grad * scale_w.view(1, -1, 1)) # (N, CKK, L)
                grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation) # (N, C, H, W)
                
                # grad_weight
                q_x = at.backend.ops.lut_lookup_int8(q_x, dw_lut)
                grad_weight = torch.matmul(upstream_grad * scale_x, q_x.transpose(1, 2).contiguous()) # (N, O, CKK)
                grad_weight = grad_weight.sum(dim=0) # (O, CKK)
                grad_weight = grad_weight.view(O, C, kH, kW)
            case _:
                raise ValueError("Invalid quantization method")
        
        return grad_x, grad_weight, None, None, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None

class _conv2d_int8_custom(_conv2d_int8_base):
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_x, grad_weight, grad_bias = None, None, None
        q_x, q_w = ctx.saved_tensors
        scale_x, zero_x, scale_w, zero_w = ctx.scale_x, ctx.zero_x, ctx.scale_w, ctx.zero_w
        dx_lut, dw_lut = ctx.dx_lut, ctx.dw_lut
        
        B, O, OH, OW = ctx.output_shape
        B, C, H, W = ctx.input_shape
        kH, kW = ctx.kernel_size
        L = OH * OW
        upstream_grad = upstream_grad.view(B,O,L)
        
        if ctx.has_bias and ctx.needs_input_grad[12]:
            grad_bias = upstream_grad.sum(dim=(0, 2))
            
        
        # upstream_grad (N, O, L)
        # q_x (N, CKK, L)
        # q_w (O, CKK)
        match (ctx.x_quantizer[1], ctx.w_quantizer[2]):
            case ('symmetric', 'tensor'):
                # grad_x
                grad_x = at.backend.ops.bgemm_custom_grad_int8_dx(q_x, q_w, upstream_grad, dx_lut) * scale_w 
                grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation) # (N, C, H, W)
                
                # grad_weight
                grad_weight = at.backend.ops.bgemm_custom_grad_int8_dw(q_x, q_w, upstream_grad, dw_lut) * scale_x # (O, CKK)
                grad_weight = grad_weight.view(O, C, kH, kW)
                
            case ('symmetric', 'channel'):
                # grad_x 
                upstream_grad_scaled = upstream_grad * scale_w.view(1, O, 1)
                grad_x = at.backend.ops.bgemm_custom_grad_int8_dx(q_x, q_w, upstream_grad_scaled, dx_lut)
                grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation) # (N, C, H, W)
                
                # grad_weight
                grad_weight = at.backend.ops.bgemm_custom_grad_int8_dw(q_x, q_w, upstream_grad, dw_lut) * scale_x # (O, CKK)
                grad_weight = grad_weight.view(O, C, kH, kW)
                
            case _:
                raise ValueError("Invalid quantization method")
        
        return grad_x, grad_weight, None, None, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None

def conv2d_int8(x, weight, lut, grad, dx_lut, dw_lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax):
    
    match grad:
        case 'ste':
            return _conv2d_int8_ste.apply(x, weight, lut, grad, dx_lut, dw_lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax)
        case 'int_ste':
            return _conv2d_int8_int_ste.apply(x, weight, lut, grad, dx_lut, dw_lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax)
        case 'custom':
            return _conv2d_int8_custom.apply(x, weight, lut, grad, dx_lut, dw_lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax)
        case 'lre':
            return _conv2d_int8_lre.apply(x, weight, lut, grad, dx_lut, dw_lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax)
        case 'date':
            pass
        case _:
            raise ValueError("Invalid gradient type")


class Conv2d_int8(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 x_quantizer: tuple[str, str, str] = ('dynamic', 'symmetric', 'tensor'),
                 w_quantizer: tuple[str, str, str] = ('dynamic', 'symmetric', 'tensor'),
                 grad: str = 'ste',
                 grad_dx: torch.Tensor | None = None,
                 grad_dy: torch.Tensor | None = None,
                 bias: torch.Tensor | None = None,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1
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
        self.qmin = -128
        self.qmax = 127
        
        # lut 
        self.register_buffer('lut', lut)
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

        if self.grad == 'custom' or self.grad == 'lre':
            self.register_buffer('grad_dx', grad_dx)
            self.register_buffer('grad_dy', grad_dy)
        else:
            self.grad_dx = None
            self.grad_dy = None
            
        # leave these to be None for now, maybe will be used in the future
        self.scale_x = None
        self.zero_x = None
        self.scale_w = None
        self.zero_w = None
    
    def __repr__(self):
        return f"Conv2d_int8(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"x_quantizer={self.x_quantizer}, w_quantizer={self.w_quantizer}, grad={self.grad})"
      

    
    def forward(self, x: torch.Tensor):
            
        output = conv2d_int8(x, self.weight, self.lut, self.grad, self.grad_dx, self.grad_dy, self.x_quantizer, self.w_quantizer,
                                self.scale_x, self.zero_x, self.scale_w, self.zero_w, self.bias,
                                self.stride, self.padding, self.dilation, self.groups, self.qmin, self.qmax)
        
        return output
    
    

