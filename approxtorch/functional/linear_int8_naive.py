import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q




class linear_naive(Function):
    @staticmethod
    def forward(ctx, feature, weight, lut, bias=None):
        # input(B, in_features), 
        # weight(out_features, in_features), 
        # bias(out_features)
        # output(B, out_features)
        weight = weight.transpose(0, 1)  # (in_features, out_features)
        
        # quantization first 
        feature, scale_feature = Q.quantize_tensor(feature) # quantized still float
        weight, scale_weight = Q.quantize_tensor(weight)
        
        feature = feature.to(torch.int8)
        weight = weight.to(torch.int8)
        output = ap.ops.gemm_int8_naive(feature, weight, lut).to(torch.float)  # (in_features, B)
        
        output = output * scale_feature * scale_weight
        if bias != None:
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                output = output + bias

        return output