import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import math

from . import im2col
import approxtorch.approx_gemm as ap

def quantize_dynamic_int4_per_tensor(x: torch.Tensor, dim = (0,1,2,3)):
    with torch.no_grad():
        abs_max = torch.amax(torch.abs(x), dim=dim, keepdim=False)
        scale = abs_max / 7.5
        x = torch.round(x / scale)
        x = torch.clamp(x, -8, 7)
        return x, scale

def quantize_static_int4_per_tensor(x: torch.Tensor, scale: torch.Tensor):
    with torch.no_grad():
        x = torch.round(x / scale)
        x = torch.clamp(x, -8, 7)
        return x

def quantize_static_int4_per_channel(x: torch.Tensor, scale: torch.Tensor):
    with torch.no_grad():
        scale = scale.view(-1,1,1,1)
        x = torch.round(x / scale)
        x = torch.clamp(x, -8, 7)
        return x


class _conv2d_int4_exact(Function):
    @staticmethod
    def forward(
                feature, 
                weight, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                scale_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1
                ):
        
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)

        # quantization first 
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = quantize_dynamic_int4_per_tensor(feature)
                weight, scale_weight = quantize_dynamic_int4_per_tensor(weight)
            case ('static', 'tensor', 'tensor'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_tensor(weight, scale_weight)
            case ('static', 'tensor', 'channel'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_channel(weight, scale_weight)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # do the convolution
        # this one is the convolution with exact mulitplication
        output = F.conv2d(feature, weight, bias, stride, padding, dilation, groups)
        # output shape is (B, O, OH, OW)
        # de-quantize the output
        
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
     
        return output
    
    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
    @staticmethod
    def backward(ctx, upstream_grad):
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        
        return grad_feature, grad_weight, None, None, None, grad_bias, None, None, None, None, None
    

def conv2d_int4_exact(feature, 
                      weight, 
                      qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                      scale_feature: torch.Tensor | None = None, 
                      scale_weight: torch.Tensor | None = None, 
                      bias = None, 
                      stride: int | tuple[int, int] = 1, 
                      padding: int | tuple[int, int] = 0, 
                      dilation: int | tuple[int, int] = 1, 
                      groups: int = 1):
    return _conv2d_int4_exact.apply(feature, 
                                    weight, 
                                    qmethod, 
                                    scale_feature, 
                                    scale_weight, 
                                    bias, 
                                    stride, 
                                    padding, 
                                    dilation, 
                                    groups)

class _conv2d_int4_STE(Function):
    @staticmethod
    def forward(feature, 
                weight, 
                lut, 
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                scale_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1):
        
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        
        # quantize here
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = quantize_dynamic_int4_per_tensor(feature)
                weight, scale_weight = quantize_dynamic_int4_per_tensor(weight)
            case ('static', 'tensor', 'tensor'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_tensor(weight, scale_weight)
            case ('static', 'tensor', 'channel'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_channel(weight, scale_weight)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
            
        # im2col
        feature = im2col.conv_window(feature, (Kh, Kw), stride, padding, dilation).to(torch.int8)
        weight = im2col.conv_weight(weight).to(torch.int8)
        # feature is (B*L, CKK) and weight shape is (CKK, O)
         
        # approximate gemm
        output = ap.ops.gemm_int4(feature, weight, lut).to(torch.float)
        # output shape is (BL, O)

        # re-arrange tensor, de-quantize and add bias if exists
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)

        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
        
        if bias is not None:
            if bias.shape[0] != output.shape[1]:
                raise ValueError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
                
        return output
            

    @staticmethod
    def setup_context(ctx, input, output):
        feature, weight, _, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups = input
        ctx.save_for_backward(feature, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
    
    @staticmethod
    def backward(ctx, upstream_grad):
        # load the saved tensors
        feature, weight = ctx.saved_tensors
        grad_feature, grad_weight, grad_bias = None, None, None
        if ctx.has_bias and ctx.needs_input_grad[6]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        
        return grad_feature, grad_weight, None, None, None, None, grad_bias, None, None, None, None
    

def conv2d_int4_STE(feature, 
                    weight, 
                    lut, 
                    qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'), 
                    scale_feature: torch.Tensor | None = None, 
                    scale_weight: torch.Tensor | None = None, 
                    bias = None, 
                    stride: int | tuple[int, int] = 1, 
                    padding: int | tuple[int, int] = 0, 
                    dilation: int | tuple[int, int] = 1, 
                    groups: int = 1):
    return _conv2d_int4_STE.apply(feature, weight, lut, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups)


# conv2d use estimated gradient
class _conv2d_int4_bit(Function):
    @staticmethod
    def forward(ctx, 
                feature: torch.Tensor, 
                weight: torch.Tensor, 
                lut: torch.Tensor, 
                gradient_lut: tuple[torch.Tensor, torch.Tensor],
                qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'channel'),
                scale_feature: torch.Tensor | None = None, 
                scale_weight: torch.Tensor | None = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0, 
                dilation: int | tuple[int, int] = 1, 
                groups: int = 1):
        
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.has_bias = bias is not None
        ctx.feature_shape = feature.shape
        ctx.weight_shape = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        ctx.output_shape = (B, O, OH, OW)
        
        match qmethod:
            case ('dynamic', 'tensor', 'tensor'):
                feature, scale_feature = quantize_dynamic_int4_per_tensor(feature)
                weight, scale_weight = quantize_dynamic_int4_per_tensor(weight)
            case ('static', 'tensor', 'tensor'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_tensor(weight, scale_weight)
            case ('static', 'tensor', 'channel'):
                feature = quantize_static_int4_per_tensor(feature, scale_feature)
                weight = quantize_static_int4_per_channel(weight, scale_weight)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # im2col
        feature = im2col.conv_window(feature, (Kh, Kw), stride, padding, dilation).to(torch.int8)
        weight = im2col.conv_weight(weight).to(torch.int8)

        # save for backward 
        ctx.save_for_backward(feature, weight, gradient_lut[0], gradient_lut[1], scale_feature, scale_weight)
        output = ap.ops.gemm_int4(feature, weight, lut).to(torch.float)  # (B*L, O)
        
        # rearrange tensor
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)

        # dequantization 
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output * scale_feature * scale_weight
            case (_, 'tensor', 'channel'):
                output = output * scale_feature * scale_weight.view(1, -1, 1, 1)
        
        if bias != None:         
            if bias.shape[0] != output.shape[1]:
                raise RuntimeError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
                
        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        mat_feature, mat_weight, grad_lut_dx, grad_lut_dy, s_feature, s_weight = ctx.saved_tensors
        # 要把 grad_lut 拆成两个，一个是dx, 一个是dy
        # feature is (B*L, CKK)
        # weight is (CKK, O)
        # upstream_grad is (B, O, OH, OW)
        # grad_lut shape (256, 2)
        (B, C, H, W) = ctx.feature_shape
        (B, O, OH, OW) = ctx.output_shape
        (_, _, Kh, Kw) = ctx.weight_shape
        L = OH * OW
        grad_bias =None
        #  if bias gradient is needed
        if ctx.has_bias and ctx.needs_input_grad[7]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)

            mat_feature = mat_feature.to(torch.int)
            mat_weight = mat_weight.to(torch.int)
            
            d_feature = torch.zeros_like(mat_feature, dtype=torch.float)
            d_weight = torch.zeros_like(mat_weight, dtype=torch.float)
            
            for k in range(C*Kh*Kw):
                A_k = mat_feature[:, k] + 8
                B_k = mat_weight[k, :] + 8
                
                ga = grad_lut_dx[A_k[:, None], B_k[None, :]]
                gb = grad_lut_dy[A_k[:, None], B_k[None, :]]

                d_feature[:, k] = (upstream_grad * ga).sum(dim=1)
                d_weight[k, :] = (upstream_grad * gb).sum(dim=0)

            
            d_feature = d_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
            d_feature = torch.nn.functional.fold(d_feature, (H, W), 
                        kernel_size=(Kh, Kw), padding=ctx.padding, 
                        stride=ctx.stride, dilation=ctx.dilation) * s_weight


            # grad_weight shape is (CKK, O)
            d_weight = d_weight * s_feature
      
            d_weight = d_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw)

        return d_feature, d_weight, None, None, None, None, None, grad_bias, None, None, None, None
    

def conv2d_int4_bit(feature,
                    weight,
                    lut,
                    gradient_lut: tuple[torch.Tensor, torch.Tensor],
                    qmethod: tuple[str, str, str] = ('dynamic', 'tensor', 'tensor'),
                    scale_feature: torch.Tensor | None = None,
                    scale_weight: torch.Tensor | None = None,
                    bias = None,
                    stride: int | tuple[int, int] = 1,
                    padding: int | tuple[int, int] = 0,
                    dilation: int | tuple[int, int] = 1,
                    groups: int = 1):
    return _conv2d_int4_bit.apply(feature, weight, 
                               lut, gradient_lut, qmethod, scale_feature, scale_weight, bias, stride, padding, dilation, groups)





