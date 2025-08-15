import torch
import torch.nn.grad
import approxtorch as at
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import math
from . import im2col
from . import quantization as Q
from typing import Tuple, Union

class _depthwise_conv2d_STE(Function):
    @staticmethod
    def forward(feature, 
                weight,
                lut, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                scale_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                bias = None, 
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int]] = 0,
                dilation: Union[int, Tuple[int, int]] = 1,
                groups: int = 1
                ):
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (C, 1, KH, KW) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(KH-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(KW-1) - 1)/stride[1] + 1)
        L = OH * OW
        
        # 1. quantize 
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = Q.quantize_dynamic_int8_per_tensor(feature)
                weight, scale_weight = Q.quantize_dynamic_int8_per_tensor(weight)
            case ('dynamic', 'tensor', 'channel'):
                feature, scale_feature = Q.quantize_dynamic_int8_per_tensor(feature)
                weight, scale_weight = Q.quantize_dynamic_int8_per_channel(weight)
            case ('static', 'tensor', 'tensor'):
                if scale_feature is not None and scale_weight is not None:
                    feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
                    weight = Q.quantize_static_int8_per_tensor(weight, scale_weight)
                else:
                    raise ValueError("qparams is not provided")
            case ('static', 'tensor', 'channel'):
                if scale_feature is not None and scale_weight is not None:
                    feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
                    weight = Q.quantize_static_int8_per_channel(weight, scale_weight)
                else:
                    raise ValueError("qparams is not provided")
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # 2. im2col (different here)
        feature = torch.nn.functional.unfold(
                    feature,
                    (KH, KW),
                    stride=stride,
                    padding=padding,
                    dilation=dilation) # (B, CKK, L)
        feature = feature.view(B, C, KH*KW, L).to(torch.int8)
        weight = weight.view(C, 1, KH*KW).to(torch.int8)
        # feature shape (B, C, KK, L)
        # weight shape (C, 1, KK)
        
        # 3. approximate depthwise gemm
        output = at.approx_gemm.ops.depthwise_gemm_int8(feature, weight, lut)
        # output shape is (B, C, 1, L) type int32
        
        output = output.view(B, C, L).view(B, C, OH, OW)
        output = output.to(torch.float)
        
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
        
        if bias is not None:
            if bias.shape[0] != output.shape[1]:
                raise ValueError('the shape of the bias is not right')
        
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
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None 
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, 
                                                      weight, 
                                                      upstream_grad, 
                                                      stride=ctx.stride, 
                                                      padding=ctx.padding, 
                                                      dilation=ctx.dilation,
                                                      groups=ctx.groups
                                                      )
            grad_weight = torch.nn.grad.conv2d_weight(feature, 
                                                      weight.shape, 
                                                      upstream_grad, 
                                                      stride=ctx.stride, 
                                                      padding=ctx.padding, 
                                                      dilation=ctx.dilation,
                                                      groups=ctx.groups
                                                      )
            
        return grad_feature, grad_weight, None, None, None, None, grad_bias, None, None, None, None
    
def depthwise_conv2d_STE(feature,
                    weight,
                    lut,
                    qmethod: Tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                    scale_feature: torch.Tensor | None = None,
                    scale_weight: torch.Tensor | None = None,
                    bias = None,
                    stride: Union[int, Tuple[int, int]] = 1,
                    padding: Union[int, Tuple[int, int]] = 0,
                    dilation: Union[int, Tuple[int, int]] = 1,
                    groups: int = 1):
    return _depthwise_conv2d_STE.apply(
            feature, 
            weight, 
            lut, 
            qmethod, 
            scale_feature, 
            scale_weight, 
            bias, 
            stride, 
            padding, 
            dilation, 
            groups)