import torch
import torch.nn as nn
import approxtorch as at
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import approxtorch as at
import math


# approximate conv2d with learnable scales for int8
class _approx_conv2d_int8_lsq_ste(Function):
    @staticmethod
    def forward(ctx, feature, weight, lut, 
                scale_feature, scale_weight, 
                kernel_size, padding, stride, dilation, output_shape):
        
        # ========== 量化 feature ==========
        scale_feature_safe = scale_feature.abs() + 1e-8
        f_scaled = feature / scale_feature_safe
        feature_mask = (f_scaled >= -128) & (f_scaled <= 127)
        qfeature = torch.round(f_scaled).clamp(-128, 127)

        # ========== 量化 weight (per-channel) ==========
        scale_weight_safe = scale_weight.abs() + 1e-8
        sw_view = scale_weight_safe.view(-1, 1, 1, 1)
        w_scaled = weight / sw_view
        weight_mask = (w_scaled >= -128) & (w_scaled <= 127)
        qweight = torch.round(w_scaled).clamp(-128, 127)
        
        # 保存原始 shape 的张量用于 backward
        ctx.save_for_backward(
            feature, weight, 
            scale_feature_safe, scale_weight_safe, 
            qfeature, qweight,  # 保存量化后但未变形的张量
            feature_mask, weight_mask,
            f_scaled, w_scaled  # 保存用于 LSQ 梯度计算
        )
        ctx.parameters = (kernel_size, padding, stride, dilation, output_shape)
        
        # ========== im2col + approx bgemm ==========
        qfeature_int8 = qfeature.to(torch.int8)
        qweight_int8 = qweight.to(torch.int8)
        
        qfeature_col = at.approx_gemm.ops.im2col_int8(
            qfeature_int8, kernel_size[0], kernel_size[1], 
            padding[0], padding[1], stride[0], stride[1], 
            dilation[0], dilation[1]
        )
        qweight_flat = qweight_int8.view(output_shape[1], -1)

        output = at.approx_gemm.ops.approx_bgemm_int8(qfeature_col, qweight_flat, lut)
        
        # 反量化
        scale = scale_feature_safe * scale_weight_safe
        output = output * scale.view(1, -1, 1, 1)
        output = output.view(output_shape[0], output_shape[1], output_shape[2], output_shape[3])
        
        return output
    
    @staticmethod 
    def backward(ctx, upstream_grad):
        (feature, weight, 
         scale_feature, scale_weight, 
         qfeature, qweight,
         feature_mask, weight_mask,
         f_scaled, w_scaled) = ctx.saved_tensors
        kernel_size, padding, stride, dilation, output_shape = ctx.parameters
        
        grad_input = None
        grad_weight = None
        grad_sf = None
        grad_sw = None
        
        # 量化后的值 (用于 STE 反传)
        sw_view = scale_weight.view(-1, 1, 1, 1)
        qfeature_fp = qfeature * scale_feature      # [B, C_in, H, W]
        qweight_fp = qweight * sw_view              # [C_out, C_in, K, K]
        
        # ========== 1. 对 feature 的梯度 (STE) ==========
        if ctx.needs_input_grad[0]:
            grad_input_hat = torch.nn.grad.conv2d_input(
                feature.shape, qweight_fp, upstream_grad, 
                stride, padding, dilation
            )
            grad_input = grad_input_hat * feature_mask
        
        # ========== 2. 对 weight 的梯度 (STE) ==========
        if ctx.needs_input_grad[1]:
            grad_weight_hat = torch.nn.grad.conv2d_weight(
                qfeature_fp, weight.shape, upstream_grad, 
                stride, padding, dilation
            )
            grad_weight = grad_weight_hat * weight_mask
        
        # ========== 3. 对 scale_feature 的梯度 (LSQ) ==========
        if ctx.needs_input_grad[3]:
            if grad_input_hat is None:
                grad_input_hat = torch.nn.grad.conv2d_input(
                    feature.shape, qweight_fp, upstream_grad, 
                    stride, padding, dilation
                )
            
            # 范围内: qfeature - f_scaled
            # 范围外: clamp(f_scaled, -128, 127) 就是边界值 -128 或 127
            d_qf_dsf = torch.where(
                feature_mask,
                qfeature - f_scaled,
                torch.clamp(f_scaled, -128, 127)  # 下溢=-128, 上溢=127
            )
            
            g_sf = 1.0 / (feature.numel() * 127.0) ** 0.5
            grad_sf = (grad_input_hat * d_qf_dsf).sum() * g_sf
        
        # ========== 4. 对 scale_weight 的梯度 (LSQ, per-channel) ==========
        if ctx.needs_input_grad[4]:
            if grad_weight_hat is None:
                grad_weight_hat = torch.nn.grad.conv2d_weight(
                    qfeature_fp, weight.shape, upstream_grad, 
                    stride, padding, dilation
                )
            
            d_qw_dsw = torch.where(
                weight_mask,
                qweight - w_scaled,
                torch.clamp(w_scaled, -128, 127)
            )
            
            g_sw = 1.0 / (weight[0].numel() * 127.0) ** 0.5
            grad_sw = (grad_weight_hat * d_qw_dsw).sum(dim=(1, 2, 3)) * g_sw
            
        return grad_input, grad_weight, None, grad_sf, grad_sw, None, None, None, None, None
            
        
def approx_conv2d_int8_lsq_ste(feature, weight, lut, scale_feature, scale_weight, kernel_size, padding, stride, dilation, output_shape):
    return _approx_conv2d_int8_lsq_ste.apply(feature, weight, lut, scale_feature, scale_weight, kernel_size, padding, stride, dilation, output_shape)
        

# approximate conv2d for int8 
# input is float, weight is float, output is float
def approx_conv2d(feature, weight, lut, scale_feature, scale_weight, kernel_size, padding, stride, dilation, output_shape):
    qfeature = feature / scale_feature
    qfeature = torch.round(qfeature).clamp(-128, 127).to(torch.int8)
    
    
    qweight = weight / scale_weight.view(-1, 1, 1, 1)
    qweight = torch.round(qweight).clamp(-128, 127).to(torch.int8)
    
    
    qfeature = at.approx_gemm.ops.im2col_int8(qfeature, kernel_size[0], kernel_size[1], 
            padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1])
    qweight = qweight.view(output_shape[1], -1)
    
    output = at.approx_gemm.ops.approx_bgemm_int8(qfeature, qweight, lut).to(torch.float)
    scale = scale_feature * scale_weight
    output = output * scale.view(1, -1, 1, 1)
    output = output.view(output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    return output
    

# learnable scale version of _conv2d_BN_qat for int8 approximate mulitplier
class _conv2d_bn_qat_int8_ste(Function):
    @staticmethod
    def forward(ctx, 
                feature,  # float feature,
                weight,        # float weight
                lut,
                scale_feature, # which is also the scale activation
                scale_weight,  # scale of the weight
                scale_output,  # for re-quantization int32 -> int8
                stride, padding, dilation,
                running_mean,  # BN mean  
                running_var,   # BN var
                momentum,      # BN momentum
                eps,           # BN eps
                gamma,         # BN gamma  trainable parameter weight
                beta,          # BN beta  trainable parameter bias
                is_training   = True, # if training, we need to update the mean and var
            ):
        
        (B, C, H, W) = feature.shape
        (O, C, Kh, Kw) = weight.shape
        OH = math.floor((H + 2*padding[0] - dilation[0]*(Kh-1) - 1)/stride[0] + 1)
        OW = math.floor((W + 2*padding[1] - dilation[1]*(Kw-1) - 1)/stride[1] + 1)
        output_shape = (B, O, OH, OW)
        kernel_size = (Kh, Kw)
        # 1. fold the BN into weights 
        if is_training:
            with torch.no_grad():
                first_conv_output = approx_conv2d(feature, weight, 
                                scale_feature, scale_weight, lut, kernel_size, 
                                padding, stride, dilation, output_shape)
                batch_mean = first_conv_output.mean(dim=(0, 2, 3))
                batch_var = first_conv_output.var(dim=(0, 2, 3), unbiased=False)
                batch_std = torch.sqrt(batch_var + eps)
                running_mean.copy_(momentum * batch_mean + (1 - momentum) * running_mean)
                running_var.copy_(momentum * batch_var + (1 - momentum) * running_var)
            
            weight_factor = gamma / torch.sqrt(running_var + eps)
            weight_fold = weight * weight_factor.view(-1, 1, 1, 1)
            fold_weight_scale = scale_weight * weight_factor
            output = approx_conv2d_int8_lsq_ste(feature, weight_fold, scale_feature, fold_weight_scale, lut, kernel_size, 
                            padding, stride, dilation, output_shape)
            
            
            bias_fold = (beta - gamma * batch_mean / batch_std).view(1, -1, 1, 1)
            output_factor = (torch.sqrt(running_var + eps) / batch_std).view(1, -1, 1, 1)
            
            output = output * output_factor + bias_fold

        else:
            r_std = torch.sqrt(running_var + eps)
            w_factor = gamma / r_std
            fold_weight = weight * w_factor.view(-1, 1, 1, 1)
            fold_weight_scale = scale_weight * w_factor
            output = approx_conv2d(feature, fold_weight, scale_feature, fold_weight_scale, lut, kernel_size, 
                            padding, stride, dilation, output_shape)
            bias_fold = (beta - gamma * running_mean / r_std).view(1, -1, 1, 1)
            output_factor = (r_std / torch.sqrt(running_var + eps)).view(1, -1, 1, 1)
            output = output * output_factor + bias_fold
            
        # re-quantization
        output = torch.round(output / scale_output)
        output = torch.clamp(output, -128, 127)
        output = output * scale_output
        return output
            
def conv2d_bn_qat_int8_ste(feature, weight, lut, scale_feature, scale_weight, scale_output, stride, padding, dilation, running_mean, running_var, momentum, eps, gamma, beta, is_training):
    return _conv2d_bn_qat_int8_ste.apply(feature, weight, lut, scale_feature, scale_weight, scale_output, stride, padding, dilation, running_mean, running_var, momentum, eps, gamma, beta, is_training)

class Conv2dBN_qat_int8(torch.nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                kernel_size: int | tuple[int, int], 
                lut: torch.Tensor,
                bias: torch.Tensor | None = None,
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] = 0,
                dilation: int | tuple[int, int] = 1,
                # BN parameters
                eps=1e-05, 
                momentum=0.1,
                # module args
                freeze_bn=False,
                # forward_mode: str = "approx"  # should be 'approx' or 'slow' only
            ):
        super().__init__()
        self.freeze_bn = freeze_bn if self.training else True
        self.register_buffer('lut', lut)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.eps = eps
        self.momentum = momentum
        
        # weight of Conv2d,
        # including float weight, weight scale, feature scale, and output scale
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.scale_feature = torch.nn.Parameter(torch.empty([]), requires_grad=True)
        self.scale_weight = torch.nn.Parameter(torch.empty([out_channels]), requires_grad=True)
        self.scale_output = torch.nn.Parameter(torch.empty([]), requires_grad=True)
        
        # BN parameters
        self.gamma = torch.nn.Parameter(torch.empty([out_channels]), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty([out_channels]), requires_grad=True)
        self.register_buffer('running_mean', torch.zeros([out_channels]))
        self.register_buffer('running_var', torch.ones([out_channels]))
        
    def __repr__(self):
        return f"Conv2dBN_QAT_int8(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " \
            f"dilation={self.dilation})"
        
    def forward(self, x):
        return conv2d_bn_qat_int8_ste(x, 
                self.weight, 
                self.lut, 
                self.scale_feature, 
                self.scale_weight, 
                self.scale_output, 
                self.stride, 
                self.padding, 
                self.dilation, 
                self.running_mean, self.running_var, 
                self.momentum, 
                self.eps, 
                self.gamma, 
                self.beta, 
                self.training)

            
        
        
        