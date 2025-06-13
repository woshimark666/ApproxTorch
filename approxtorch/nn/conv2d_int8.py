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

class _conv2d_int8_STE(Function):
    @staticmethod
    def forward(feature, 
                weight, 
                lut, 
                qmethod: Tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                qparams: Tuple[torch.Tensor, torch.Tensor] | None = None, 
                bias = None, 
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int]] = 0,
                dilation: Union[int, Tuple[int, int]] = 1
                ):
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        
        scale_feature, scale_weight = None, None
        # quantize here
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = Q.quantize_dynamic_int8_per_tensor(feature)
                weight, scale_weight = Q.quantize_dynamic_int8_per_tensor(weight)
            case ('dynamic', 'tensor', 'channel'):
                feature, scale_feature = Q.quantize_dynamic_int8_per_tensor(feature)
                weight, scale_weight = Q.quantize_dynamic_int8_per_channel(weight)
            case ('static', 'tensor', 'tensor'):
                if qparams is not None:
                    scale_feature, scale_weight = qparams
                    feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
                    weight = Q.quantize_static_int8_per_tensor(weight, scale_weight)
                else:
                    raise ValueError("qparams is not provided")
            case ('static', 'tensor', 'channel'):
                if qparams is not None:
                    scale_feature, scale_weight = qparams
                    feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
                    weight = Q.quantize_static_int8_per_channel(weight, scale_weight)
                else:
                    raise ValueError("qparams is not provided")
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
            
        # im2col
        feature = im2col.conv_window(feature, (Kh, Kw), stride, padding, dilation).to(torch.int8)
        weight = im2col.conv_weight(weight).to(torch.int8)
        # feature is (B*L, CKK) and weight shape is (CKK, O)
         
        # approximate gemm
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)
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
        feature, weight, _, _, _, bias, stride, padding, dilation = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
    
    @staticmethod
    def backward(ctx, upstream_grad):
        # load the saved tensors
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            
        return grad_feature, grad_weight, None, None, None, grad_bias, None, None, None
    
def conv2d_int8_STE(feature,
                    weight,
                    lut,
                    qmethod: Tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                    qparams: Tuple[torch.Tensor, torch.Tensor] | None = None,
                    bias = None,
                    stride: Union[int, Tuple[int, int]] = 1,
                    padding: Union[int, Tuple[int, int]] = 0,
                    dilation: Union[int, Tuple[int, int]] = 1):
    return _conv2d_int8_STE.apply(feature, weight, lut, qmethod, qparams, bias, stride, padding, dilation)
    


# conv2d use estimated gradient
# class conv2d_est(Function):
#     @staticmethod
#     def forward(ctx, feature, weight, lut, gradient_lut, bias=None, stride:int=1, padding:int=0, dilation:int=1):
#         (B, C, H, W) = feature.shape
#         (O, C, Kh, Kw) = weight.shape
#         ctx.stride = stride
#         ctx.padding = padding
#         ctx.dilation = dilation
#         ctx.has_bias = bias is not None
#         ctx.feature_shape = feature.shape
#         ctx.weight_shape = weight.shape
#         OH = math.floor((H + 2*padding - dilation*(Kh-1) - 1)/stride + 1)
#         OW = math.floor((W + 2*padding - dilation*(Kw-1) - 1)/stride + 1)
#         L = OH * OW
#         ctx.output_shape = (B, O, OH, OW)
#         feature, scale_feature = Q.quantize_tensor(feature)
#         weight, scale_weight = Q.quantize_tensor(weight)
#         feature = im2col.conv_window(feature, Kh, stride=stride, padding=padding, dilation=dilation).to(torch.int8)
#         # feature is (B*L, CKK) and int8 type
#         weight = im2col.conv_weight(weight).to(torch.int8)
#         ctx.save_for_backward(feature, weight, gradient_lut)
#         ctx.scale_feature = scale_feature
#         ctx.scale_weight = scale_weight
#         # weight is (CKK, O) and int8 type
#         output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (B*L, O)
        
#         output = output * scale_feature * scale_weight
        
#         output = output.view(B, L, O)
#         output = output.transpose(1, 2) # (B, O, L)
#         output = output.contiguous()
#         output = output.view(B, O, OH, OW)
        
#         if bias != None:
            
#             if bias.shape[0] != output.shape[1]:
#                 raise RuntimeError('the shape of the bias is not right')
#             else:
#                 bias = bias.view(1, -1, 1, 1)
#                 output = output + bias
                
#         return output
    
#     @staticmethod
#     def backward(ctx, upstream_grad):
#         mat_feature, mat_weight, grad_lut = ctx.saved_tensors
#         # feature is (B*L, CKK)
#         # weight is (CKK, O)
#         # upstream_grad is (B, O, OH, OW)
#         # grad_lut is (255*255, 2)
#         (B, C, H, W) = ctx.feature_shape
#         (B, O, OH, OW) = ctx.output_shape
#         (_, _, Kh, Kw) = ctx.weight_shape
#         L = OH * OW
#         grad_feature, grad_weight, grad_bias = None, None, None
#         #  if bias gradient is needed
#         if ctx.has_bias and ctx.needs_input_grad[4]:
#             grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
#         if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
#             upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)
#             grad_feature, grad_weight = ap.ops.gemm_int8_gradient(mat_feature, mat_weight, upstream_grad, grad_lut)
#             # grad_feature [BL, CKK], grad_weight [CKK, O]
#             grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
#             grad_feature =  torch.nn.functional.fold(grad_feature, (H, W), 
#                         kernel_size=(Kh, Kw), padding=ctx.padding, 
#                         stride=ctx.stride, dilation=ctx.dilation) * ctx.scale_weight

#             grad_weight = grad_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw) * ctx.scale_feature
            
#         return grad_feature, grad_weight, None, None, grad_bias, None, None, None
