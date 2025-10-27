import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import math
from . import im2col
import approxtorch.approx_gemm as ap

def quantize_dynamic_int4_per_tensor(x: torch.Tensor, dim = (0,1,2,3)):
    with torch.no_grad():
        abs_max = torch.amax(torch.abs(x), dim=dim, keepdim=False)
        scale = abs_max / 7.5
        x = torch.round(x / scale)
        x = torch.clamp(x, -8, 7)
        return x, scale

def quantize_static_int4_per_tensor(x: torch.Tensor, scale: torch.Tensor):
    with torch.no_grad():
        x = torch.round(x / scale)
        x = torch.clamp(x, -8, 7)
        return x

def quantize_static_int4_per_channel(x: torch.Tensor, scale: torch.Tensor):
    with torch.no_grad():
        scale = scale.view(-1,1,1,1)
        x = torch.round(x / scale)
        x = torch.clamp(x, -8, 7)
        return x


class _conv2d_int4_exact(Function):
    @staticmethod
    def forward(
                feature, 
                weight, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                scale_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1
                ):
        
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)

        # quantization first 
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = quantize_dynamic_int4_per_tensor(feature)
                weight, scale_weight = quantize_dynamic_int4_per_tensor(weight)
            case ('static', 'tensor', 'tensor'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_tensor(weight, scale_weight)
            case ('static', 'tensor', 'channel'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_channel(weight, scale_weight)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # do the convolution
        # this one is the convolution with exact mulitplication
        output = F.conv2d(feature, weight, bias, stride, padding, dilation, groups)
        # output shape is (B, O, OH, OW)
        # de-quantize the output
        
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
     
        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
    @staticmethod
    def backward(ctx, upstream_grad):
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        
        return grad_feature, grad_weight, None, None, None, grad_bias, None, None, None, None, None
    

def conv2d_int4_exact(feature, 
                      weight, 
                      qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                      scale_feature: torch.Tensor | None = None, 
                      scale_weight: torch.Tensor | None = None, 
                      bias = None, 
                      stride: int | tuple[int, int] = 1, 
                      padding: int | tuple[int, int] = 0, 
                      dilation: int | tuple[int, int] = 1, 
                      groups: int = 1):
    return _conv2d_int4_exact.apply(feature, 
                                    weight, 
                                    qmethod, 
                                    scale_feature, 
                                    scale_weight, 
                                    bias, 
                                    stride, 
                                    padding, 
                                    dilation, 
                                    groups)

class _conv2d_int4_STE(Function):
    @staticmethod
    def forward(feature, 
                weight, 
                lut, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                scale_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1):
        
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        
        # quantize here
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = quantize_dynamic_int4_per_tensor(feature)
                weight, scale_weight = quantize_dynamic_int4_per_tensor(weight)
            case ('static', 'tensor', 'tensor'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_tensor(weight, scale_weight)
            case ('static', 'tensor', 'channel'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_channel(weight, scale_weight)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
            
        # im2col
        feature = im2col.conv_window(feature, (Kh, Kw), stride, padding, dilation).to(torch.int8)
        weight = im2col.conv_weight(weight).to(torch.int8)
        # feature is (B*L, CKK) and weight shape is (CKK, O)
         
        # approximate gemm
        output = ap.ops.gemm_int4(feature, weight, lut).to(torch.float)
        # output shape is (BL, O)

        # re-arrange tensor, de-quantize and add bias if exists
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)

        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
        
        if bias is not None:
            if bias.shape[0] != output.shape[1]:
                raise ValueError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
                
        return output
            

    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, _, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
    
    @staticmethod
    def backward(ctx, upstream_grad):
        # load the saved tensors
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None
        if ctx.has_bias and ctx.needs_input_grad[6]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        
        return grad_feature, grad_weight, None, None, None, None, grad_bias, None, None, None, None
    

def conv2d_int4_STE(feature, 
                    weight, 
                    lut, 
                    qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                    scale_feature: torch.Tensor | None = None, 
                    scale_weight: torch.Tensor | None = None, 
                    bias = None, 
                    stride: int | tuple[int, int] = 1, 
                    padding: int | tuple[int, int] = 0, 
                    dilation: int | tuple[int, int] = 1, 
                    groups: int = 1):
    return _conv2d_int4_STE.apply(feature, weight, lut, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups)