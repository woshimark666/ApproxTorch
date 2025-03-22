from typing import Any, Tuple
import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q




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
        
# wrap the linear for functional use
def linear_int8(feature, weight, lut, bias=None):
    return linear.apply(feature, weight, lut, bias)