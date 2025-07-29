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

class Conv2DInt8STE(Function):
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
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            
        return grad_feature, grad_weight, None, None, None, None, grad_bias, None, None, None, None
    
def conv2d_int8_STE(feature,
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
    return Conv2DInt8STE.apply(feature, weight, lut, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups)
    


# conv2d use estimated gradient
class Conv2DInt8EST(Function):
    @staticmethod
    def forward(ctx, 
                feature: torch.Tensor, 
                weight: torch.Tensor, 
                lut: torch.Tensor, 
                gradient_lut: torch.Tensor,
                qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'channel'),
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
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.has_bias = bias is not None
        ctx.feature_shape = feature.shape
        ctx.weight_shape = weight.shape
        OH = math.floor((H + 2*padding - dilation*(Kh-1) - 1)/stride + 1)
        OW = math.floor((W + 2*padding - dilation*(Kw-1) - 1)/stride + 1)
        L = OH * OW
        ctx.output_shape = (B, O, OH, OW)
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
                    raise ValueError("scale is not provided")
            case ('static', 'tensor', 'channel'):
                if scale_feature is not None and scale_weight is not None:
                    feature = Q.quantize_static_int8_per_tensor(feature, scale_feature)
                    weight = Q.quantize_static_int8_per_channel(weight, scale_weight)
                else:
                    raise ValueError("scale is not provided")
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # im2col
        feature = im2col.conv_window(feature, (Kh, Kw), stride, padding, dilation).to(torch.int8)
        weight = im2col.conv_weight(weight).to(torch.int8)

        # save for backward 
        ctx.save_for_backward(feature, weight, scale_feature, scale_weight, gradient_lut)
        # weight is (CKK, O) and int8 type
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (B*L, O)
        
        # rearrange tensor
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)

        # dequantization 
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
        
        if bias != None:         
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
                
        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        mat_feature, mat_weight, scale_feature, scale_weight, grad_lut = ctx.saved_tensors
        # 要把 grad_lut 拆成两个，一个是dx, 一个是dy
        grad_lut_dx, grad_lut_dy = torch.unbind(grad_lut, dim=1)
        # feature is (B*L, CKK)
        # weight is (CKK, O)
        # upstream_grad is (B, O, OH, OW)
        # grad_lut is (255*255, 2)
        (B, C, H, W) = ctx.feature_shape
        (B, O, OH, OW) = ctx.output_shape
        (_, _, Kh, Kw) = ctx.weight_shape
        L = OH * OW
        grad_feature, grad_weight, grad_bias = None, None, None
        #  if bias gradient is needed
        if ctx.has_bias and ctx.needs_input_grad[4]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)
            grad_feature, grad_weight = ap.ops.gemm_int8_gradient(mat_feature, mat_weight, grad_lut_dx, grad_lut_dy)
            # grad_feature is for computing feature gradient, shape is (CKK, O)
            # grad_weight is for computing weight gradient, shape is (BL, CKK)
            # upstream_grad shape is changed to (BL, O)
            grad_feature = upstream_grad.matmul(grad_feature.t()) # output shape (BL, CKK)
            grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
            grad_feature =  torch.nn.functional.fold(grad_feature, (H, W), 
                        kernel_size=(Kh, Kw), padding=ctx.padding, 
                        stride=ctx.stride, dilation=ctx.dilation) * scale_weight
            # here the grad_feature shape is (B, C, H, W)


            grad_weight = grad_weight.t().matmul(upstream_grad) # output shape (CKK, O)
            grad_weight = grad_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw) * scale_feature

            
        return grad_feature, grad_weight, None, None, None, None, None, grad_bias, None, None, None, None
    

def conv2d_int8_EST(feature,
                    weight,
                    lut,
                    gradient_lut,
                    qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'channel'),
                    scale_feature: torch.Tensor | None = None,
                    scale_weight: torch.Tensor | None = None,
                    bias = None,
                    stride: Union[int, Tuple[int, int]] = 1,
                    padding: Union[int, Tuple[int, int]] = 0,
                    dilation: Union[int, Tuple[int, int]] = 1,
                    groups: int = 1):
    return Conv2DInt8EST.apply(feature, weight, 
                               lut, gradient_lut, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups)
    