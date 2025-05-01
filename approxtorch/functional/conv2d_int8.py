from typing import Any, Tuple
import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q
from . import im2col
import math

class conv2d(Function):
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
        ctx.gradient_lut = gradient_lut
        OH = math.floor((H + 2*padding - dilation*(Kh-1) - 1)/stride + 1)
        OW = math.floor((W + 2*padding - dilation*(Kw-1) - 1)/stride + 1)
        L = OH * OW
        ctx.output_shape = (B, O, OH, OW)
        feature, scale_feature = Q.quantize_tensor(feature)
        weight, scale_weight = Q.quantize_tensor(weight)
        feature = im2col.conv_window(feature, Kh, stride=stride, padding=padding, dilation=dilation).to(torch.int8)
        # feature is (B*L, CKK) and int8 type
        weight = im2col.conv_weight(weight).to(torch.int8)
        ctx.save_for_backward(feature, weight)
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
        feature, weight = ctx.saved_tensors
        # feature is (B*L, CKK)
        # weight is (CKK, 0)
        # scale_feature and scale_weight are float number
        # upstream_grad is (B, O, OH, OW)
        (B, C, H, W) = ctx.feature_shape
        (B, O, OH, OW) = ctx.output_shape
        (_, _, Kh, Kw) = ctx.weight_shape
        L = OH * OW
        grad_feature, grad_weight, grad_bias = None, None, None
        #  if bias gradient is needed
        if ctx.has_bias and ctx.needs_input_grad[4]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            d_feature = torch.zeros_like(feature, dtype=torch.float32) # (BL, C*Kh*Kw)
            d_weight = torch.zeros_like(weight, dtype=torch.float32)  # (C*Kh*Kw, O)
            upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)
            
            # turn upstream_grad to (BL, O)
            for i in range(B*L):
                for j in range(O):
                    for k in range(C*Kh*Kw):
                        a = int(feature[i, k].item())
                        b = int(weight[k, j].item())
                        index = 255 * (a+127) + b + 127
                        df = ctx.gradient_lut[index]
                        df_da = df[0]
                        df_db = df[1]
                        
                        d_feature[i, k] += df_da * upstream_grad[i, j]
                        d_weight[k, j] += df_db * upstream_grad[i, j]
            
            d_feature = d_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous() # (B, C*Kh*Kw, L)
            d_feature = torch.nn.functional.fold(d_feature, (H, W), 
                        kernel_size=(Kh, Kw), padding=ctx.padding, 
                        stride=ctx.stride, dilation=ctx.dilation) * ctx.scale_weight

            # now is the gradient for weight
            # d_weight (C*Kh*Kw, O) 
            d_weight = d_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw) * ctx.scale_feature
            # d_weight is (O, C, Kh, Kw)
            
        return d_feature, d_weight, None, None, grad_bias, None, None, None
    
    # @staticmethod
    # def setup_context(ctx, input, output):
    #     feature, weight, lut, bias, stride, padding, dilation = input
    #     ctx.save_for_backward(feature, weight)
    #     ctx.has_bias = bias is not None
    #     ctx.stride = stride
    #     ctx.padding = padding
    #     ctx.dilation = dilation
        
    # # this is ste straight through estimator
    # @staticmethod
    # def backward(ctx, upstream_grad):
    #     # load the saved tensors
    #     feature, weight = ctx.saved_tensors
    #     grad_feature, grad_weight, grad_bias = None, None, None

    #     # if bias gradient is needed
    #     if ctx.has_bias and ctx.needs_input_grad[4]:
    #         grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        
    #     if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
    #         grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
    #         grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            
    #     return grad_feature, grad_weight, None, None, grad_bias, None, None, None

    # not ste here 
    # @staticmethod
    # def backward(ctx, upstream_grad):
    #     # load the saved tensors
    #     feature, weight = ctx.saved_tensors
    #     # feature is (B, CKK, L)
    #     # weight is (O, CKK)
    #     # scale_feature and scale_weight are float number
    #     # upstream_grad is (B, O, OH, OW)
    #     grad_feature, grad_weight, grad_bias = None, None, None
        
    #     # if bias gradient is needed
    #     if ctx.has_bias and ctx.needs_input_grad[4]:
    #         grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        
    #     if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
    #         upstream_grad = upstream_grad.view(ctx.B, ctx.O, ctx.L) # upstream_grad is (B, O, L)
    #         weight_tmp = weight.transpose(0, 1).to(torch.float) * ctx.scale_weight # weight_tmp is (CKK, O) and float
    #         grad_feature = torch.matmul(weight_tmp, upstream_grad)  # grad_feature is (B, CKK, L)
    #         grad_feature = torch.nn.functional.fold(grad_feature, (ctx.H, ctx.W), kernel_size=(ctx.K, ctx.K), padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation)
    #         # the grad_feature is (B, C, H, W) and finish for grad_feature
    #         # basiclly the grad_feature is the weights

    #         # now is the gradient for weight
    #         feature_tmp = feature.transpose(1,2).to(torch.float) * ctx.scale_feature # feature tmp is (B, L, CKK) upstream_grad is (B, O, L)
    #         grad_weight = torch.matmul(upstream_grad, feature_tmp) # grad_weight is (B, O, CKK)
    #         grad_weight = grad_weight.sum(0).view(ctx.O, ctx.C, ctx.K, ctx.K) # grad weight is (O, C, K, K)
            
    #     return grad_feature, grad_weight, None, None, grad_bias, None, None, None

    


# cov2d with int input and output
# class int_conv2d(Function):
#     @staticmethod
#     def forward(ctx, feature, weight, lut, bias=None, stride=1, padding=0, dilation=1):
#         H = feature.shape[2]
#         W = feature.shape[3]
#         B = feature.shape[0]
#         O = weight.shape[0]
#         K = weight.shape[2]
#         OH = math.floor((H + 2*padding - dilation*(K-1) - 1)/stride + 1)
#         OW = math.floor((W + 2*padding - dilation*(K-1) - 1)/stride + 1)
#         L = OH * OW
#         # quantization first 
#         # input is int8, to test the correctness
#         # feature = feature.to(torch.float)
#         feature = im2col.conv_window(feature, K, stride=stride, padding=padding, dilation=dilation).to(torch.int8)
#         # feature becomes (B, CKK, L) and int8 type
        
#         weight = im2col.conv_weight(weight).to(torch.int8)
#         # weight becomes (O, CKK) and int8 type
        
#         output = ap.batch_gemm_int8(feature, weight, lut).to(torch.float)
#         output = output.view(B, O, OH, OW)
        
#         return output
        


# warp the conv2d for functional use
def conv2d_int8(feature, weight, lut, gradient_lut, bias=None, stride:int = 1, padding:int = 0, dilation:int = 1):
    return conv2d.apply(feature, weight, lut, gradient_lut, bias, stride, padding, dilation)