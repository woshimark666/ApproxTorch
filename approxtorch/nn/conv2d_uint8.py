import torch
from . import quantization as Q
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import math
from . import im2col

class _conv2d_uint8_STE(Function):
    @staticmethod
    def forward(feature, 
                weight, 
                lut, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                qparams: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None, 
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1
                ):
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        
        scale_feature, scale_weight, zero_point_feature, zero_point_weight = None, None, None, None
        # 1. quantize 
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature, zero_point_feature = Q.quantize_dynamic_uint8_per_tensor(feature)
                weight, scale_weight, zero_point_weight = Q.quantize_dynamic_uint8_per_tensor(weight)
            case ('dynamic', 'tensor', 'channel'):
                feature, scale_feature, zero_point_feature = Q.quantize_dynamic_uint8_per_tensor(feature)
                weight, scale_weight, zero_point_weight = Q.quantize_dynamic_uint8_per_channel(weight)
            case ('static', 'tensor', 'tensor'):
                if qparams is not None:
                    scale_feature, zero_point_feature, scale_weight, zero_point_weight = qparams
                    feature = Q.quantize_static_uint8_per_tensor(feature, scale_feature, zero_point_feature)
                    weight = Q.quantize_static_uint8_per_tensor(weight, scale_weight, zero_point_weight)
                else:
                    raise ValueError("qparams is not provided")
            case ('static', 'tensor', 'channel'):
                if qparams is not None:
                    scale_feature, zero_point_feature, scale_weight, zero_point_weight = qparams
                    feature = Q.quantize_static_uint8_per_tensor(feature, scale_feature, zero_point_feature)
                    weight = Q.quantize_static_uint8_per_channel(weight, scale_weight, zero_point_weight)
                else:
                    raise ValueError("qparams is not provided")
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # 2. im2col
        feature = im2col.conv_window(feature, (Kh, Kw), stride, padding, dilation)
        weight = im2col.conv_weight(weight)
        
        # 3. approximate gemm  # test phase
        output = torch.matmul(feature, weight)
        # the output shape is (BL, O)
        
        # 4. de-quantize
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output - zero_point_feature * weight.sum(dim=0, keepdim=True) - \
                            zero_point_weight * feature.sum(dim=1, keepdim=True) + \
                                C*Kh*Kw*zero_point_feature*zero_point_weight
            case (_, 'tensor', 'channel'):
                output = output - zero_point_feature * weight.sum(dim=0, keepdim=True) - \
                            zero_point_weight.unsqueeze(0) * feature.sum(dim=1, keepdim=True) + \
                                C*Kh*Kw*zero_point_feature*zero_point_weight
        
        output = output * scale_feature * scale_weight    
        
        # 5. reshape the output tensor
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)
        
        # 6. add bias
        if bias is not None:
            if bias.shape[0] != output.shape[1]:
                raise ValueError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
    
        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, _, _, _, bias, stride, padding, dilation = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
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
            
        return grad_feature, grad_weight, None, None, None, grad_bias, None, None, None
    
    
def conv2d_uint8_STE(feature,
                    weight,
                    lut,
                    qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                    qparams: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                    bias = None,
                    stride: int | tuple[int, int] = 1,
                    padding: int | tuple[int, int] = 0,
                    dilation: int | tuple[int, int] = 1):
    return _conv2d_uint8_STE.apply(feature, weight, lut, qmethod, qparams, bias, stride, padding, dilation)