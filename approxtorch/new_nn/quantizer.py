import torch
from torch.autograd import Function
import math



import torch
from torch.autograd import Function

class LSQQuantize_int(Function):
    @staticmethod
    def forward(ctx, x, scale, Qn: int, Qp: int, per_channel: bool, ch_axis: int):
        # alpha: scalar or shape [C]
        # 保证 alpha > 0
        # alpha_safe = alpha.clamp_min(eps)

        if per_channel:
            # reshape alpha to broadcast over x
            shape = [1] * x.dim()
            shape[ch_axis] = -1
            a = scale.view(shape)
        else:
            a = scale

        x_div = x / a
        q = x_div.round().clamp(Qn, Qp)
        q = q.to(torch.int8)

        # mask for clamp derivative (inside range)
        m = (x_div >= Qn) & (x_div <= Qp)

        # 保存 backward 需要的数据
        ctx.save_for_backward(x_div, q, m, scale)
        ctx.Qn = Qn
        ctx.Qp = Qp
        ctx.per_channel = per_channel
        ctx.ch_axis = ch_axis

        return q

    @staticmethod
    def backward(ctx, grad_y):
        x_div, q, m, scale = ctx.saved_tensors
        Qp = ctx.Qp
        per_channel = ctx.per_channel
        ch_axis = ctx.ch_axis

        # dx: STE + clamp mask
        grad_x = grad_y * m.to(grad_y.dtype)

        # d_alpha: sum( grad_y * ( q - x/alpha * m ) ) * grad_scale
        term = (q - x_div * m.to(x_div.dtype))  # elementwise
        g_alpha_elem = grad_y * term

        if per_channel:
            # 对除 channel 轴以外的维度求和
            reduce_dims = [d for d in range(g_alpha_elem.dim()) if d != ch_axis]
            grad_alpha = g_alpha_elem.sum(dim=reduce_dims)

            # N per channel
            N = g_alpha_elem.numel() // g_alpha_elem.size(ch_axis)
            grad_scale = (1.0 / (N * Qp) ** 0.5)
            grad_alpha = grad_alpha * grad_scale
        else:
            grad_alpha = g_alpha_elem.sum()
            N = g_alpha_elem.numel()
            grad_scale = (1.0 / (N * Qp) ** 0.5)
            grad_alpha = grad_alpha * grad_scale

        # 其余输入无梯度
        return grad_x, grad_alpha, None, None, None, None, None


class LSQQuantizer_int8(Function):
    @staticmethod
    def forward(ctx, x, scale, per_channel: bool, ch_axis: int, Qn: int, Qp: int):
        # Qn, Qp 分别是上界和下界
        
        # 1. 保存变量供 backward 使用
        ctx.save_for_backward(x, scale)
        
        # 2. 量化逻辑 (和之前一样)
        x_div_s = x / scale
        q_x = torch.clamp(torch.round(x_div_s), Qn, Qp)
        q_x = q_x.to(torch.int8)
        return q_x
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     return None, None, None, None, None, None

def lsq_quantize_int8(x, scale, per_channel: bool, ch_axis: int, Qn: int, Qp: int):
    return LSQQuantizer_int8.apply(x, scale, per_channel, ch_axis, Qn, Qp)