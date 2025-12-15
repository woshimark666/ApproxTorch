import torch
from . import quantization as Q
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
import approxtorch as at

# 先把前向做好，后向先放一下
class _conv2d_uint8_custom(Function):
    @staticmethod
    def forward(
                ctx,
                feature, 
                weight, 
                lut, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                scale_feature: torch.Tensor | None = None,
                zero_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                zero_weight: torch.Tensor | None = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1
            ):
        
        # 0 pair the input parameters
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        # 1 quantization
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                qfeature, scale_feature, zero_feature = Q.quantize_dynamic_uint8_per_tensor(feature)
                qweight, scale_weight, zero_weight = Q.quantize_dynamic_uint8_per_tensor(weight)
            case ('dynamic', 'tensor', 'channel'):
                qfeature, scale_feature, zero_feature = Q.quantize_dynamic_uint8_per_tensor(feature)
                qweight, scale_weight, zero_weight = Q.quantize_dynamic_uint8_per_channel(weight)
            case ('static', 'tensor', 'tensor'):
                if scale_feature is not None and zero_feature is not None and scale_weight is not None and zero_weight is not None:
                    qfeature = Q.quantize_static_uint8_per_tensor(feature, scale_feature, zero_feature)
                    qweight = Q.quantize_static_uint8_per_tensor(weight, scale_weight, zero_weight)
                else:
                    raise ValueError("scale or zero point is not provided")
            case ('static', 'tensor', 'channel'):
                if scale_feature is not None and zero_feature is not None and scale_weight is not None and zero_weight is not None:
                    qfeature = Q.quantize_static_uint8_per_tensor(feature, scale_feature, zero_feature)
                    qweight = Q.quantize_static_uint8_per_channel(weight, scale_weight, zero_weight)
                else:
                    raise ValueError("scale or zero point is not provided")
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        qfeature = qfeature.to(torch.uint8)
        qweight = qweight.to(torch.uint8)
        
        qfeature = at.approx_gemm.ops.im2col_uint8(qfeature, 
                                weight.shape[2], weight.shape[3], 
                                padding[0], padding[1], 
                                stride[0], stride[1], 
                                dilation[0], dilation[1])
        qweight = qweight.view(weight.shape[0], -1)
        # here qfeature shape is (B,CKK,L)
        # qweight shape is (O,CKK)
        
        # perfom batched GEMM
        output = at.approx_gemm.ops.approx_bgemm_uint8(qfeature, qweight, lut)
        # output shape is (B,O,L)
        output = output.view(B, O, OH, OW)
        # change shapse to (B, O, OH, OW)
        
        output = output.to(torch.float)
        qweight = qweight.to(torch.float)
        qfeature = qfeature.to(torch.float)
        match qmethod:
            case (_, 'tensor', 'tensor'):
                # we set the zero_W to be 0.
                # then the whole equation is much eaiser.
                term1 = zero_feature * qweight.sum(dim=(1)).view(1, -1, 1, 1)
                output = output - term1
                output = output * scale_feature * scale_weight 
            case (_, 'tensor', 'channel'):
                pass 
            # TODO: add when I have time
            
        # 3. add bias if needed
        if bias is not None:
            if bias.shape[0] != output.shape[1]:
                raise ValueError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
        
        return output
    
    
def conv2d_uint8_custom(feature, weight, lut, qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), scale_feature: torch.Tensor | None = None, zero_feature: torch.Tensor | None = None, scale_weight: torch.Tensor | None = None, zero_weight: torch.Tensor | None = None, bias = None, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1):
    return _conv2d_uint8_custom.apply(feature, weight, lut, qmethod, scale_feature, zero_feature, scale_weight, zero_weight, bias, stride, padding, dilation, groups)
        
class Conv2d_uint8_custom(torch.nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int], 
            lut: torch.Tensor,
            qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
            scale_feature: torch.Tensor | None = None,
            zero_feature: torch.Tensor | None = None,
            scale_weight: torch.Tensor | None = None,
            zero_weight: torch.Tensor | None = None,
            bias: bool | torch.Tensor = True,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            dilation: int | tuple[int, int] = 1,
            groups: int = 1):
        super().__init__()
        self.register_buffer('lut', lut)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.qmethod = qmethod
        self.bias = bias
        self.has_bias = bias is not None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.frozen_scale = True
    
        match qmethod[0]:
            case 'static':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=False)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=False)
                self.zero_feature = torch.nn.Parameter(zero_feature, requires_grad=False)
                self.zero_weight = torch.nn.Parameter(zero_weight, requires_grad=False)
            case 'dynamic':
                self.scale_feature = None
                self.scale_weight = None
            case 'trainable':
                self.scale_feature = torch.nn.Parameter(scale_feature, requires_grad=True)
                self.scale_weight = torch.nn.Parameter(scale_weight, requires_grad=True)
            case _:
                raise ValueError("Invalid quantization method")
            
        if isinstance(self.bias, torch.Tensor):
            self.bias = torch.nn.Parameter(self.bias)
            self.has_bias = True
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.has_bias = True
        elif bias == False or bias == None:
            self.bias = None
            self.has_bias = False
        else:
            raise ValueError("Invalid bias type")
        
        
    def __repr__(self):
        return f"Conv2d_uint8_custom(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
            f"kernel_size={self.kernel_size}, qmethod={self.qmethod}, " \
            f"bias={self.has_bias}, stride={self.stride}, padding={self.padding}, " \
            f"dilation={self.dilation}, groups={self.groups}, freeze_scales={self.frozen_scale})"
    def updata_scale(self, x, weight, qmethod):
        max_feature = torch.max(x)
        min_feature = torch.min(x)
        new_scale_feature = (max_feature - min_feature) / 255.
        new_zero_feature = - torch.round(min_feature / new_scale_feature)
        if qmethod[2] == 'channel':
            max_weight = torch.amax(weight, dim=(1,2,3), keepdim=False)
            min_weight = torch.amin(weight, dim=(1,2,3), keepdim=False)
            new_scale_weight = (max_weight - min_weight) / 255.
            new_zero_weight = - torch.round(min_weight / new_scale_weight)
        elif qmethod[2] == 'tensor':
            max_weight = torch.max(weight)
            min_weight = torch.min(weight)
            new_scale_weight = (max_weight - min_weight) / 255.
            new_zero_weight = - torch.round(min_weight / new_scale_weight)
        
        new_scale_feature = 0.95 * self.scale_feature + 0.05 * new_scale_feature
        new_zero_feature = 0.95 * self.zero_feature + 0.05 * new_zero_feature
        new_scale_weight = 0.95 * self.scale_weight + 0.05 * new_scale_weight
        new_zero_weight = 0.95 * self.zero_weight + 0.05 * new_zero_weight
        
        with torch.no_grad():
            self.scale_feature.copy_(new_scale_feature)
            self.scale_weight.copy_(new_scale_weight)
            self.zero_feature.copy_(new_zero_feature)
            self.zero_weight.copy_(new_zero_weight)

    def freeze_scale(self):
        self.frozen_scale = True
    
    def unfreeze_scale(self):
        self.frozen_scale = False
        
    def forward(self, x):
        if not self.frozen_scale and self.qmethod[0] == 'static':
            self.updata_scale(x, self.weight, self.qmethod)
        return conv2d_uint8_custom(x, 
                            self.weight, 
                            self.lut, 
                            self.qmethod, 
                            self.scale_feature, 
                            self.zero_feature, 
                            self.scale_weight, 
                            self.zero_weight, 
                            self.bias, 
                            self.stride, 
                            self.padding, 
                            self.dilation, 
                            self.groups)