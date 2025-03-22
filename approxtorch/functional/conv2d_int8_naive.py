import torch
from torch.autograd import Function
import approxtorch.approx_gemm as ap
import torch.nn.grad
from . import quantization as Q
from . import im2col
import math

class conv2d_naive(Function):
    @staticmethod
    def forward(ctx, feature, weight, lut, bias=None, stride:int=1, padding:int=0, dilation:int=1):
        (B, C, H, W) = feature.shape
        (O, C, K, K) = weight.shape
        OH = math.floor((H + 2*padding - dilation*(K-1) - 1)/stride + 1)
        OW = math.floor((W + 2*padding - dilation*(K-1) - 1)/stride + 1)
        L = OH * OW
        feature, scale_feature = Q.quantize_tensor(feature)
        weight, scale_weight = Q.quantize_tensor(weight)
        feature = im2col.conv_window(feature, K, stride=stride, padding=padding, dilation=dilation).to(torch.int8)
        # feature is (B*L, CKK) and int8 type
        weight = im2col.conv_weight(weight).to(torch.int8)
        # weight is (CKK, O) and int8 type
        output = ap.ops.gemm_int8_naive(feature, weight, lut).to(torch.float)  # (B*L, O)
        
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