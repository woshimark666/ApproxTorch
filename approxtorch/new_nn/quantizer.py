import torch
from torch.autograd import Function
import math
import torch.nn as nn


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
    def forward(ctx, x, scale, per_channel: bool, ch_axis: int = 0, Qn: int = -128, Qp: int = 127):
        # Qn, Qp 分别是上界和下界
        
        # 1. 保存变量供 backward 使用
        ctx.save_for_backward(x, scale)
        
        # 2. 量化逻辑 (和之前一样)
        if per_channel:
            shape = [1] * x.dim()
            shape[ch_axis] = -1
            a = scale.view(shape)
        else:
            a = scale
        x_div_s = x / a
        q_x = torch.clamp(torch.round(x_div_s), Qn, Qp)
        q_x = q_x.to(torch.int8)
        return q_x
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     return None, None, None, None, None, None

def lsq_quantize_int8(x, scale, per_channel: bool, ch_axis: int, Qn: int, Qp: int):
    return LSQQuantizer_int8.apply(x, scale, per_channel, ch_axis, Qn, Qp)




class LSQQuantizerFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, q_min, q_max, g):
        x_s = x / scale
        x_q = torch.round(x_s).clamp(q_min, q_max)
        ctx.save_for_backward(x_s, x_q)
        ctx.params = (q_min, q_max, g)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        x_s, x_q = ctx.saved_tensors
        q_min, q_max, g = ctx.params
        
        mask = (x_s >= q_min) & (x_s <= q_max)
        grad_x = grad_output * mask
        
        d_xhat_ds = torch.where(mask, x_q - x_s, x_s.clamp(q_min, q_max))
        grad_scale = (grad_output * d_xhat_ds).sum() * g
        
        return grad_x, grad_scale, None, None, None


class LSQQuantizer(nn.Module):
    def __init__(self, bit_width=8, symmetric=True, per_channel=False, num_channels=1):
        super().__init__()
        self.per_channel = per_channel
        
        if symmetric:
            self.q_min = -2 ** (bit_width - 1)
            self.q_max = 2 ** (bit_width - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** bit_width - 1
        
        shape = (num_channels,) if per_channel else ()
        self.scale = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        scale = self.scale.abs() + 1e-8
        if self.per_channel:
            scale = scale.view(1, -1, *([1] * (x.dim() - 2)))
        
        g = 1.0 / (x.numel() * self.q_max) ** 0.5
        return LSQQuantizerFunc.apply(x, scale, self.q_min, self.q_max, g)


# class TrainableScaleQuantizer_int8(nn.Module):
#     """
#     可训练 scale 的 8-bit 对称量化器
#     """
    
#     def __init__(
#         self,
#         per_channel: bool = False,
#         num_channels: int = 1,
#         init_scale: float = 1.0,
#         scale_min: float = 1e-5,
#         scale_max: float = 1e3,
#         qmin: int = -127,
#         qmax: int = 127,
#     ):
#         """
#         Args:
#             per_channel: 是否使用 per-channel 量化
#             num_channels: 通道数（per_channel=True 时使用）
#             init_scale: scale 初始值
#             scale_min: scale 下界
#             scale_max: scale 上界
#         """
#         super().__init__()
        
#         self.per_channel = per_channel
#         self.num_channels = num_channels
#         self.scale_min = scale_min
#         self.scale_max = scale_max
        
#         # 8-bit 对称量化范围: [-127, 127]
#         self.qmin = qmin
#         self.qmax = qmax
        
#         # 初始化可训练的 scale 参数
#         if per_channel:
#             self.scale = nn.Parameter(torch.ones(num_channels) * init_scale)
#         else:
#             self.scale = nn.Parameter(torch.tensor(init_scale))
    
#     def _get_clamped_scale(self) -> torch.Tensor:
#         """获取限制在上下界内的 scale"""
#         return torch.clamp(self.scale, self.scale_min, self.scale_max)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播：量化 -> 反量化（STE）
#         """
#         scale = self._get_clamped_scale()
        
#         # 调整 scale 形状以支持广播
#         if self.per_channel and x.dim() > 1:
#             shape = [1] * x.dim()
#             shape[1] = -1  # channel 维度
#             scale = scale.view(shape)
        
#         # 量化 + 反量化
#         x_quant = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        
#         # STE: 前向用量化值，反向传原始梯度
#         return x
    
#     def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """仅量化，返回整数值和 scale"""
#         scale = self._get_clamped_scale()
        
#         if self.per_channel and x.dim() > 1:
#             shape = [1] * x.dim()
#             shape[1] = -1
#             scale_view = scale.view(shape)
#         else:
#             scale_view = scale
        
#         x_quant = torch.clamp(torch.round(x / scale_view), self.qmin, self.qmax)
#         return x_quant.to(torch.int8), scale
    
#     def dequantize(self, x_quant: torch.Tensor) -> torch.Tensor:
#         """反量化"""
#         scale = self._get_clamped_scale()
        
#         if self.per_channel and x_quant.dim() > 1:
#             shape = [1] * x_quant.dim()
#             shape[1] = -1
#             scale = scale.view(shape)
        
#         return x_quant.float() * scale
    
#     def init_from_data(self, x: torch.Tensor):
#         """根据数据初始化 scale"""
#         with torch.no_grad():
#             if self.per_channel and x.dim() > 1:
#                 dims = [i for i in range(x.dim()) if i != 1]
#                 max_val = torch.amax(torch.abs(x), dim=dims)
#             else:
#                 max_val = torch.max(torch.abs(x))
            
#             scale = max_val / self.qmax
#             self.scale.data = torch.clamp(scale, self.scale_min, self.scale_max)
    
#     def extra_repr(self) -> str:
#         return f'per_channel={self.per_channel}, scale_range=[{self.scale_min}, {self.scale_max}]'


# # 测试
# if __name__ == "__main__":
#     print("8-bit 对称量化器测试\n")
    
#     # Per-tensor 测试
#     quantizer = TrainableScaleQuantizer(scale_min=1e-4, scale_max=100)
#     x = torch.randn(2, 4) * 10
#     quantizer.init_from_data(x)
    
#     print(f"输入:\n{x}")
#     print(f"Scale: {quantizer.scale.item():.4f} (范围: [{quantizer.scale_min}, {quantizer.scale_max}])")
#     print(f"量化输出:\n{quantizer(x)}")
    
#     # 梯度测试
#     x.requires_grad = True
#     loss = quantizer(x).sum()
#     loss.backward()
#     print(f"\nScale 梯度: {quantizer.scale.grad.item():.4f}")