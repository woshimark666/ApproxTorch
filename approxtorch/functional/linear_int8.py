from typing import Any, Tuple
import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q



# linear using the straight through estimator STE
class linear(Function):
    @staticmethod
    def forward(feature, weight, lut, bias=None):
        # input(B, in_features), 
        # weight(out_features, in_features), 
        # bias(out_features)
        # output(B, out_features)
        weight = weight.transpose(0, 1).contiguous()  # (in_features, out_features)
        
        # quantization first 
        feature, scale_feature = Q.quantize_tensor(feature) # quantized still float
        weight, scale_weight = Q.quantize_tensor(weight)
        

        feature = feature.to(torch.int8)
        weight = weight.to(torch.int8)
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (in_features, B)
        output = output * scale_feature * scale_weight
        if bias != None:
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                output = output + bias

        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, lut, bias = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
    
    @staticmethod
    def backward(ctx, upstream_grad):
        feature, weight = ctx.saved_tensors
        
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = upstream_grad.sum(dim=0)
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.matmul(upstream_grad, weight)
            grad_weight = torch.matmul(upstream_grad.transpose(0, 1), feature)
        
        return grad_feature, grad_weight, None, grad_bias
        



class linear_est(Function):
    @staticmethod
    def forward(ctx, feature, weight, lut, gradient_lut, bias=None):
        # input(B, in_features), 
        # weight(out_features, in_features), 
        # bias(out_features)
        # output(B, out_features)
        weight = weight.transpose(0, 1).contiguous()  # (in_features, out_features)
        
        # quantization first 
        feature, scale_feature = Q.quantize_tensor(feature) # quantized still float
        weight, scale_weight = Q.quantize_tensor(weight)
        

        feature = feature.to(torch.int8)
        weight = weight.to(torch.int8)
        ctx.save_for_backward(feature, weight, gradient_lut)
        ctx.scale_feature = scale_feature
        ctx.scale_weight = scale_weight
        ctx.has_bias = bias is not None
        ctx.feature_shape = feature.shape
        ctx.weight_shape = weight.shape
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (in_features, B)
        output = output * scale_feature * scale_weight
        if bias != None:
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                output = output + bias

        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        feature, weight, gradient_lut = ctx.saved_tensors
        (B, in_features) = ctx.feature_shape
        (out_features, in_features) = ctx.weight_shape

        grad_feature = grad_weight = None
        if ctx.has_bias and ctx.needs_input_grad[4]:
            grad_bias = upstream_grad.sum(dim=0)
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature, grad_weight = ap.ops.gemm_int8_gradient(feature, weight, upstream_grad, gradient_lut)
            # grad_feature [B, in_features], grad_weight [in_features, out_features]
            grad_feature = grad_feature * ctx.scale_weight
            grad_weight = grad_weight.transpose(0, 1).contiguous() * ctx.scale_feature
        
        return grad_feature, grad_weight, None, None, grad_bias
 
# linear using the straight through estimator STE and with defined threshold
class linear_T(Function):
    @staticmethod
    def forward(feature, weight, lut, T_feature, T_weight, bias=None):
        # input(B, in_features), 
        # weight(out_features, in_features), 
        # bias(out_features)
        # output(B, out_features)
        weight = weight.transpose(0, 1).contiguous()  # (in_features, out_features)
        
        # quantization first 
        feature, scale_feature = Q.quantize_by_threshold(feature, T_feature)
        weight, scale_weight = Q.quantize_by_threshold(weight, T_weight)
        

        feature = feature.to(torch.int8)
        weight = weight.to(torch.int8)
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (in_features, B)
        output = output * scale_feature * scale_weight
        if bias != None:
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                output = output + bias

        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, _, _, _, bias = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
    
    @staticmethod
    def backward(ctx, upstream_grad):
        feature, weight = ctx.saved_tensors
        
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=0)
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.matmul(upstream_grad, weight)
            grad_weight = torch.matmul(upstream_grad.transpose(0, 1), feature)
        
        return grad_feature, grad_weight, None, None, None, grad_bias
 


class linear_est_T(Function):
    @staticmethod
    def forward(ctx, feature, weight, lut, gradient_lut, T_feature, T_weight, bias=None):
        # input(B, in_features), 
        # weight(out_features, in_features), 
        # bias(out_features)
        # output(B, out_features)
        weight = weight.transpose(0, 1).contiguous()  # (in_features, out_features)
        
        # quantization first 
        feature, scale_feature = Q.quantize_by_threshold(feature, T_feature) # quantized still float
        weight, scale_weight = Q.quantize_by_threshold(weight, T_weight)
        

        feature = feature.to(torch.int8)
        weight = weight.to(torch.int8)
        ctx.save_for_backward(feature, weight, gradient_lut)
        ctx.scale_feature = scale_feature
        ctx.scale_weight = scale_weight
        ctx.has_bias = bias is not None
        ctx.feature_shape = feature.shape
        ctx.weight_shape = weight.shape
        output = ap.ops.gemm_int8(feature, weight, lut).to(torch.float)  # (in_features, B)
        output = output * scale_feature * scale_weight
        if bias != None:
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                output = output + bias

        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        feature, weight, gradient_lut = ctx.saved_tensors
        (B, in_features) = ctx.feature_shape
        (out_features, in_features) = ctx.weight_shape

        grad_feature = grad_weight = None
        if ctx.has_bias and ctx.needs_input_grad[6]:
            grad_bias = upstream_grad.sum(dim=0)
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature, grad_weight = ap.ops.gemm_int8_gradient(feature, weight, upstream_grad, gradient_lut)
            # grad_feature [B, in_features], grad_weight [in_features, out_features]
            grad_feature = grad_feature * ctx.scale_weight
            grad_weight = grad_weight.transpose(0, 1).contiguous() * ctx.scale_feature
        
        return grad_feature, grad_weight, None, None, None, None, grad_bias

    
def linear_int8_est(feature, weight, lut, gradient_lut, bias=None):
    return linear_est.apply(feature, weight, lut, gradient_lut, bias)


def linear_int8(feature, weight, lut, bias=None):
    return linear.apply(feature, weight, lut, bias)

def linear_int8_T(feature, weight, lut, T_feature, T_weight, bias=None):
    return linear_T.apply(feature, weight, lut, T_feature, T_weight, bias)

def linear_int8_est_T(feature, weight, lut, gradient_lut, T_feature, T_weight, bias=None):
    return linear_est_T.apply(feature, weight, lut, gradient_lut, T_feature, T_weight, bias)