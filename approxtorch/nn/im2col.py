import torch
import approxtorch as at
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair


class _im2col_uint8(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride = 1, padding = 0, dilation = 1):
        ctx.input_shape = input.shape
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)

        # im2col for uint8
        output = at.backend.ops.im2col_uint8(feature, kernel_size, stride, padding, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [B, C*kH*kW, L] (float)
        return: grad_input [B, C, H, W]
        
        im2col 的反向 = col2im = F.fold
        """
        B, C, H, W = ctx.input_shape

        grad_input = F.fold(
            grad_output,
            output_size=(H, W),
            kernel_size=ctx.kernel_size,
            dilation=ctx.dilation,
            padding=ctx.padding,
            stride=ctx.stride,
        )

        return grad_input, None, None, None, None
    
    

class _im2col_int8(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride = 1, padding = 0, dilation = 1):
        ctx.input_shape = input.shape
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)

        # im2col for uint8
        output = at.backend.ops.im2col_int8(feature, kernel_size, stride, padding, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [B, C*kH*kW, L] (float)
        return: grad_input [B, C, H, W]
        
        im2col 的反向 = col2im = F.fold
        """
        B, C, H, W = ctx.input_shape

        grad_input = F.fold(
            grad_output,
            output_size=(H, W),
            kernel_size=ctx.kernel_size,
            dilation=ctx.dilation,
            padding=ctx.padding,
            stride=ctx.stride,
        )

        return grad_input, None, None, None, None
    
    
    
def im2col_uint8(feature, kernel_size, stride = 1, padding = 0, dilation = 1):
    return _im2col_uint8.apply(feature, kernel_size, stride, padding, dilation)

def im2col_int8(feature, kernel_size, stride = 1, padding = 0, dilation = 1):
    return _im2col_int8.apply(feature, kernel_size, stride, padding, dilation)