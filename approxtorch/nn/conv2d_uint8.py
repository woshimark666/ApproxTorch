import torch
from . import quantization as Q
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
import approxtorch as at

class _conv2d_uint8(Function):
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
                grad: str = 'ste',
                grad_data = None,
                bias = None, 
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1
                ):
        
        # 0 collect tensor shape info
        stride   = _pair(stride)
        padding  = _pair(padding)
        dilation = _pair(dilation)
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        L = OH * OW
        if grad == 'ste':
            ctx.save_for_backward(feature, weight)
        # 1. quantization 
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
        
        # 2. im2col
        qfeature = F.unfold(qfeature, (Kh, Kw), stride=stride, padding=padding, dilation=dilation) # (B, CKK, L)
        qfeature = qfeature.transpose(1, 2).contiguous().view(-1, C*Kh*Kw) # (B*L, CKK)
        qweight = qweight.view(O, -1) # (O, CKK)
        qweight = qweight.transpose(1, 0).contiguous() # (CKK, O)
        # feature is (B*L, CKK) and weight shape is (CKK, O)
        
        qfeature = qfeature.to(torch.uint8)
        qweight = qweight.to(torch.uint8)
        
        match grad:
            case 'lre' | 'custom':
                ctx.qmethod = qmethod
                ctx.feature_shape = (B, C, H, W)
                ctx.weight_shape = (O, C, Kh, Kw)
                ctx.output_shape = (B, O, OH, OW)
                ctx.save_for_backward(qfeature, qweight, scale_feature, scale_weight, zero_feature, \
                    zero_weight, grad_data[0], grad_data[1])
            
            case 'half_cutsom':
                ctx.qmethod = qmethod
                ctx.feature_shape = (B, C, H, W)
                ctx.weight_shape = (O, C, Kh, Kw)
                ctx.output_shape = (B, O, OH, OW)
                ctx.save_for_backward(weight, qfeature, qweight, scale_feature, zero_feature, grad_data[1])
                
        ctx.has_bias = bias is not None
        ctx.grad = grad
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        # 3. approximate gemm  # test phase
        output = at.approx_gemm.ops.gemm_uint8(qfeature, qweight, lut).to(torch.float)
        # the output shape is (BL, O)
        
        qfeature = qfeature.to(torch.float)
        qweight = qweight.to(torch.float)
        # 4. de-quantize
        match qmethod:
            case (_, 'tensor', 'tensor'):
                output = output - zero_feature * qweight.sum(dim=0, keepdim=True) - \
                            zero_weight * qfeature.sum(dim=1, keepdim=True) + \
                                C*Kh*Kw*zero_feature*zero_weight
                output = output * scale_feature * scale_weight    
            case (_, 'tensor', 'channel'):
                output = output - zero_feature * qweight.sum(dim=0, keepdim=True) - \
                            zero_weight.unsqueeze(0) * qfeature.sum(dim=1, keepdim=True) + \
                                C*Kh*Kw*zero_feature*zero_weight
                output = output * scale_feature * scale_weight.view(1, -1)
            case _:
                raise ValueError(f"Invalid quantization method: {qmethod}")
        
        # 5. reshape the output tensor
        output = output.view(B, L, O)
        output = output.transpose(1, 2) # (B, O, L)
        output = output.contiguous()
        output = output.view(B, O, OH, OW)
        
        # 6. add bias if needed
        if bias is not None:
            if bias.shape[0] != output.shape[1]:
                raise ValueError('the shape of the bias is not right')
            else:
                bias = bias.view(1, -1, 1, 1)
                output = output + bias
    
        return output
    
        
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_feature, grad_weight, grad_bias = None, None, None
        if ctx.has_bias and ctx.needs_input_grad[10]:
            grad_bias = upstream_grad.sum(dim=(0, 2, 3))
        match ctx.grad:
            case 'ste':
                feature, weight = ctx.saved_tensors
                grad_feature = torch.nn.grad.conv2d_input(feature.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
                grad_weight = torch.nn.grad.conv2d_weight(feature, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
            case 'lre':
                (B, C, H, W) = ctx.feature_shape
                (O, C, Kh, Kw) = ctx.weight_shape
                (B, O, OH, OW) = ctx.output_shape
                L = OH * OW
                mat_feature, mat_weight, scale_feature, scale_weight, zero_feature, zero_weight, \
                    grad_lut_dx, grad_lut_dy = ctx.saved_tensors
                upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)
                grad_feature, grad_weight = at.approx_gemm.ops.gemm_uint8_gradient(
                        mat_feature, mat_weight, grad_lut_dx, grad_lut_dy)
                # grad_feature for computing feature gradient, shape is (CKK, O)
                # grad_weight for computing weight gradient, shape is (BL, CKK)
                
                if ctx.qmethod[1:] == ('tensor', 'channel'):
                    # compute grad_feature: 
                    grad_feature = scale_weight.view(1, -1) * grad_feature - (zero_weight / scale_feature).view(1, -1) # (CKK, O)
                    grad_feature = upstream_grad.matmul(grad_feature.t()) # (BL, CKK)
                    grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
                    grad_feature = torch.nn.functional.fold(grad_feature, (H, W), 
                                kernel_size=(Kh, Kw), padding=ctx.padding, 
                                stride=ctx.stride, dilation=ctx.dilation)
                    # compute grad_weight: 
                    # upstream_grad (BL, O)
                    grad_weight = scale_feature * grad_weight # (BL, CKK)
                    grad_weight = upstream_grad.t().matmul(grad_weight) - (zero_feature / scale_weight).view(-1, 1) # (0, CKK)
                    grad_weight = grad_weight.contiguous().view(O, C, Kh, Kw)
                    
                elif ctx.qmethod[1:] == ('tensor', 'tensor'):
                    # compute grad_feature:
                    grad_feature = scale_weight * grad_feature - zero_weight / scale_feature #(CKK, O)
                    grad_feature = upstream_grad.matmul(grad_feature.t()) # (BL, CKK)
                    grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
                    grad_feature = torch.nn.functional.fold(grad_feature, (H, W), 
                                kernel_size=(Kh, Kw), padding=ctx.padding, 
                                stride=ctx.stride, dilation=ctx.dilation)
                    
                    # compute grad_weight:
                    grad_weight = scale_feature * grad_weight - zero_feature / scale_weight # (BL, CKK)
                    grad_weight = upstream_grad.t().matmul(grad_weight) # (0, CKK)
                    grad_weight = grad_weight.contiguous().view(O, C, Kh, Kw)
                    
            case 'custom':
                (B, C, H, W) = ctx.feature_shape
                (O, C, Kh, Kw) = ctx.weight_shape
                (B, O, OH, OW) = ctx.output_shape
                L = OH * OW
                mat_feature, mat_weight, scale_feature, scale_weight, zero_feature, zero_weight, \
                grad_lut_dx, grad_lut_dy = ctx.saved_tensors
                upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)
                # mat_feature (BL, CKK), mat_weight (CKK, O)
      
                if ctx.qmethod[1:] == ('tensor', 'channel'):
                    grad_feature, grad_weight = at.approx_gemm.ops.gemm_custom_grad_uint8_tc(mat_feature, mat_weight, upstream_grad, grad_lut_dx, grad_lut_dy, scale_feature, zero_feature, scale_weight, zero_weight)
                    grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
                    grad_feature = torch.nn.functional.fold(grad_feature, (H, W), 
                                kernel_size=(Kh, Kw), padding=ctx.padding, 
                                stride=ctx.stride, dilation=ctx.dilation)
                    grad_weight = grad_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw)
                    
                elif ctx.qmethod[1:] == ('tensor', 'tensor'):
                    grad_feature, grad_weight = at.approx_gemm.ops.gemm_custom_grad_uint8_tt(mat_feature, mat_weight, upstream_grad, grad_lut_dx, grad_lut_dy, scale_feature, zero_feature, scale_weight, zero_weight)
                    # grad_feature shape (BL, CKK), grad_weight shape (CKK, O)
                    grad_feature = grad_feature.view(B, L, C*Kh*Kw).transpose(1, 2).contiguous()
                    grad_feature = torch.nn.functional.fold(grad_feature, (H, W), 
                                kernel_size=(Kh, Kw), padding=ctx.padding, 
                                stride=ctx.stride, dilation=ctx.dilation)
                    grad_weight = grad_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw)
            
            case 'half_custom':
                (B, C, H, W) = ctx.feature_shape
                (O, C, Kh, Kw) = ctx.weight_shape
                (B, O, OH, OW) = ctx.output_shape
                L = OH * OW
                weight, qfeature, qweight, scale_feature, zero_feature, grad_lut_dy = ctx.saved_tensors
                
                # use ste to compute dL/dx
                grad_feature = torch.nn.grad.conv2d_input(ctx.feature_shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
                
                # use custom gradient to compute dL/dw
                upstream_grad = upstream_grad.view(B, O, L).transpose(1,2).contiguous().view(B*L, O) # (BL, O)
                grad_weight = at.approx_gemm.ops.gemm_custom_grad_uint8_dw_only(qfeature, qweight, upstream_grad, \
                    grad_lut_dy, scale_feature, zero_feature)
                grad_weight = grad_weight.transpose(0, 1).contiguous().view(O, C, Kh, Kw)
                
            case _:
                raise ValueError(f"Invalid gradient method: {ctx.grad}")
            
        return grad_feature, grad_weight, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None
    
    
def conv2d_uint8(feature,
                weight,
                lut,
                qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                scale_feature: torch.Tensor | None = None,
                zero_feature: torch.Tensor | None = None,
                scale_weight: torch.Tensor | None = None,
                zero_weight: torch.Tensor | None = None,
                grad: str = 'ste',
                grad_data = None,
                bias = None,
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                groups: int = 1):
    return _conv2d_uint8.apply(feature, 
                            weight, 
                            lut, 
                            qmethod, 
                            scale_feature, zero_feature, 
                            scale_weight, zero_weight, 
                            grad, grad_data, 
                            bias, 
                            stride, padding, dilation, groups)

