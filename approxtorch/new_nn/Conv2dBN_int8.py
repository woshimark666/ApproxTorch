import torch
from .quantizer import lsq_quantize_int8
from approxtorch.approx_gemm.ops import approx_bgemm_int8, im2col_int8
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import math
# static quantization only 
# tensor level quant for activation and chnnel level for weight
class _conv2d_bn_fake_int8(Function):
    @staticmethod
    def forward(
        ctx, 
        feature,
        weight,
        lut,
        scale_feature,
        scale_weight,
        scale_activation,
        bias,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1
    ):
        
        # 先不考虑训练的事情，我们先做前向。
        # 0 pair the input parameters
        # get some shape info
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        
        # 1. quantization
        qfeature = lsq_quantize_int8(feature, scale_feature, False, 0, -128, 127)
        qweight = lsq_quantize_int8(weight, scale_weight, True, 0, -128, 127)
        
        # 2 im2col, 
        # qfeature (B, C, H, W), qweight (O, C, Kh, Kw), 已经是int8了
        qfeature = im2col_int8(qfeature, Kh, Kw, padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1])
        qweight = qweight.view(O, -1)
        # qfeature (B, CKK, L), qweight (O, CKK)
        
        # 3. batched approx gemm
        output = approx_bgemm_int8(qfeature, qweight, lut)
        output = output.view(B, O, OH, OW)
        # output shape is (B, O, OH, OW)
        
        
        # 4. de-quantization
        output = output.to(torch.float)
        output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
        # output shape is (B, O, OH, OW)

        # 5. add bias
        output = output + bias.view(1, -1, 1, 1)
        # output = torch.nn.functional.leaky_relu(output, negative_slope=leaky_relu_k)
        # 6. re-quantization
        output = torch.round(output / scale_activation)
        output = torch.clamp(output, -128, 127)
        output = output * scale_activation
        
        return output
    
    
class Conv2dBN_fake_int8(torch.nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int], 
            lut: torch.Tensor,
            scale_feature: torch.Tensor | None = None,
            scale_weight: torch.Tensor | None = None,
            scale_activation: torch.Tensor | None = None,
            bias: torch.Tensor | None = None,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            dilation: int | tuple[int, int] = 1,
            groups: int = 1,):
        
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        
        if scale_feature is not None:
            self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=False)
        else:
            self.scale_feature = torch.nn.Parameter(torch.empty([]), requires_grad=False)
            
        if scale_weight is not None:
            self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=False)
        else:
            self.scale_weight = torch.nn.Parameter(torch.empty([out_channels]), requires_grad=False)
            
        if scale_activation is not None:
            self.scale_activation = torch.nn.Parameter(scale_activation, requires_grad=False)
        else:
            self.scale_activation = torch.nn.Parameter(torch.empty([]), requires_grad=False)
        
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)
        else:
            self.bias = torch.nn.Parameter(torch.empty([out_channels]))
            
            
    def __repr__(self):
        return f"Conv2dBN_fake_int8(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " \
            f"dilation={self.dilation}, groups={self.groups})"
        
    def forward(self, x):
        return _conv2d_bn_fake_int8.apply(x, 
                            self.weight, 
                            self.lut, 
                            self.scale_feature, 
                            self.scale_weight, 
                            self.scale_activation,
                            self.bias, 
                            self.stride, 
                            self.padding, 
                            self.dilation, 
                            self.groups)