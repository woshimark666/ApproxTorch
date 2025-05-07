from typing import Any, Tuple
import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q
from . import im2col
import math

# conv2d use estimated gradient
class conv2d_est(Function):
    @staticmethod
    def forward(ctx, feature, weight, lut, gradient_lut, bias=None, stride:int=1, padding:int=0, dilation:int=1):
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
        feature, scale_feature = Q.quantize_tensor(feature)
        weight, scale_weight = Q.quantize_tensor(weight)
        feature = im2col.conv_window(feature, Kh, stride=stride, padding=padding, dilation=dilation).to(torch.int8)
        # feature is (B*L, CKK) and int8 type
        weight = im2col.conv_weight(weight).to(torch.int8)
        ctx.save_for_backward(feature, weight, gradient_lut)
        ctx.scale_feature = scale_feature
        ctx.scale_weight = scale_weight
        # weight is (CKK, O) and int8 type
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (B*L, O)
        
        output = output * scale_feature * scale_weight
        
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)
        
        if bias != None:
            
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
                
        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        mat_feature, mat_weight, grad_lut = ctx.saved_tensors
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
            grad_feature, grad_weight = ap.ops.gemm_int8_gradient(mat_feature, mat_weight, upstream_grad, grad_lut)
            # grad_feature [BL, CKK], grad_weight [CKK, O]
            grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
            grad_feature =  torch.nn.functional.fold(grad_feature, (H, W), 
                        kernel_size=(Kh, Kw), padding=ctx.padding, 
                        stride=ctx.stride, dilation=ctx.dilation) * ctx.scale_weight

            grad_weight = grad_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw) * ctx.scale_feature
            
        return grad_feature, grad_weight, None, None, grad_bias, None, None, None



# conv2d use the straight through estimator
class conv2d(Function):
    @staticmethod
    def forward(feature, weight, lut, bias=None, stride:int=1, padding:int=0, dilation:int=1):
        (B, C, H, W) = feature.shape
        (O, C, _, K) = weight.shape
        OH = math.floor((H + 2*padding - dilation*(K-1) - 1)/stride + 1)
        OW = math.floor((W + 2*padding - dilation*(K-1) - 1)/stride + 1)
        L = OH * OW
        feature, scale_feature = Q.quantize_tensor(feature)
        weight, scale_weight = Q.quantize_tensor(weight)
        feature = im2col.conv_window(feature, K, stride=stride, padding=padding, dilation=dilation).to(torch.int8)
        # feature is (B*L, CKK) and int8 type
        weight = im2col.conv_weight(weight).to(torch.int8)
        # weight is (CKK, O) and int8 type
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (B*L, O)
        
        output = output * scale_feature * scale_weight
        
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)
        
        if bias != None:
            
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
                
        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, lut, bias, stride, padding, dilation = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
    # this is ste straight through estimator
    @staticmethod
    def backward(ctx, upstream_grad):
        # load the saved tensors
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None

        # if bias gradient is needed
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            
        return grad_feature, grad_weight, None, grad_bias, None, None, None



# warp the conv2d for functional use
def conv2d_int8(feature, weight, lut, gradient_lut, bias=None, stride:int = 1, padding:int = 0, dilation:int = 1):
    return conv2d.apply(feature, weight, lut, bias, stride, padding, dilation)

def conv2d_int8_est(feature, weight, lut, gradient_lut, bias=None, stride:int = 1, padding:int = 0, dilation:int = 1):
    return conv2d_est.apply(feature, weight, lut, gradient_lut, bias, stride, padding, dilation)