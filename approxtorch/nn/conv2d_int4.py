from typing import Any, Tuple
import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q
from . import im2col
import math
from typing import Tuple, Union
from torch.nn.modules.utils import _pair
    
class _conv2d_int4_STE(Function):
    @staticmethod
    def forward(feature, 
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
                feature, scale_feature = Q.quantize_dynamic_int4_per_tensor(feature)
                weight, scale_weight = Q.quantize_dynamic_int4_per_tensor(weight)
            # case ('dynamic', 'tensor', 'channel'):
            #     feature, scale_feature = Q.quantize_dynamic_int8_per_tensor(feature)
            #     weight, scale_weight = Q.quantize_dynamic_int8_per_channel(weight)
            # case ('static', 'tensor', 'tensor'):
            #     if scale_feature is not None and scale_weight is not None:
            #         feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
            #         weight = Q.quantize_static_int8_per_tensor(weight, scale_weight)
            #     else:
            #         raise ValueError("qparams is not provided")
            # case ('static', 'tensor', 'channel'):
            #     if scale_feature is not None and scale_weight is not None:
            #         feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
            #         weight = Q.quantize_static_int8_per_channel(weight, scale_weight)
            #     else:
            #         raise ValueError("qparams is not provided")
            
            # case ('trainable', 'tensor', 'tensor'):
            #     if scale_feature is not None and scale_weight is not None:
            #         feature = Q.quantize_trainable_int8_per_tensor(feature, scale_feature)
            #         weight = Q.quantize_trainable_int8_per_tensor(weight, scale_weight)
            #     else:
            #         raise ValueError("trainable scales are not provided")
            # case ('trainable', 'tensor', 'channel'):
            #     if scale_feature is not None and scale_weight is not None:
            #         feature = Q.quantize_trainable_int8_per_tensor(feature, scale_feature)
            #         weight = Q.quantize_trainable_int8_per_channel(weight, scale_weight)
            #     else:
            #         raise ValueError("trainable scales are not provided")
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
                    qmethod: Tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                    scale_feature: torch.Tensor | None = None,
                    scale_weight: torch.Tensor | None = None,
                    bias = None,
                    stride: Union[int, Tuple[int, int]] = 1,
                    padding: Union[int, Tuple[int, int]] = 0,
                    dilation: Union[int, Tuple[int, int]] = 1,
                    groups: int = 1):
    return _conv2d_int4_STE.apply(feature, weight, lut, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups)
    