from typing import Any, Tuple
import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q



# linear using the straight through estimator STE
class _linear_uint8_STE(Function):
    # for linear layer, only tensor level quantization is supported
    @staticmethod
    def forward(feature: torch.Tensor, 
                weight: torch.Tensor, 
                lut: torch.Tensor,
                qmethod: str = 'dynamic',
                qparams: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                bias=None
                ):
        # input(B, in_features), 
        # weight(out_features, in_features), 
        # bias(out_features)
        # output(B, out_features)
        weight = weight.transpose(0, 1).contiguous()  # (in_features, out_features)
        
        scale_feature, scale_weight = None, None
        # 1. quantization  
        match qmethod:
            case 'dynamic':
                feature, scale_feature, zero_point_feature = Q.quantize_dynamic_uint8(feature, (0,1))
                weight, scale_weight, zero_point_weight = Q.quantize_dynamic_uint8(weight, (0,1))
                
            case 'static':
                if qparams is not None:
                    scale_feature, zero_point_feature, scale_weight, zero_point_weight = qparams
                    feature = Q.quantize_static_uint8_per_tensor(feature, scale_feature, zero_point_feature)
                    weight = Q.quantize_static_uint8_per_tensor(weight, scale_weight, zero_point_weight)
                else:
                    raise ValueError("qparams is not provided")
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        feature = feature.to(torch.uint8)
        weight = weight.to(torch.uint8)
        
        # 2. approximate gemm and add compensate 
        output = ap.ops.gemm_uint8(feature, weight, lut).to(torch.float)  # (B, out_features)
        output = output - zero_point_feature * weight.sum(dim=0, keepdim=True) - \
                            zero_point_weight * feature.sum(dim=1, keepdim=True) + \
                                feature.shape[1]*zero_point_feature*zero_point_weight
        
        # 3. add bias
        if bias != None:
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                output = output + bias

        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, _, _,_, bias = input
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
        
def linear_uint8_STE(feature, 
                    weight, 
                    lut, 
                    qmethod: str = 'dynamic', 
                    qparams: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None, 
                    bias=None):
    return _linear_uint8_STE.apply(feature, weight, lut, qmethod, qparams, bias)

