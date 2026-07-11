import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules.utils import _pair
from . import quantization
from . import bgemm_uint8


def uint8_qparams(mn, mx, qmax=255):
    """由 min/max 统计导出非对称 (scale, zero_point)，zero_point 为整数值。

    先把范围 clamp 到包含 0：保证 zero_point 落在 [0, qmax]、真实 0 恰好
    在格点上（padding 才能精确表示）。mn/mx 可为标量（per-tensor）或
    [O]（per-channel），两种粒度同一套公式。
    """
    mn = torch.clamp(mn, max=0.0)
    mx = torch.clamp(mx, min=0.0)
    scale = torch.clamp((mx - mn) / qmax, min=1e-12)
    zero = torch.round(-mn / scale).clamp(0.0, float(qmax))
    return scale, zero


class Conv2d_uint8(nn.Module):
    # uint8 × uint8 近似乘法器卷积（与 Conv2d_int8 相同的 decoupled 结构：
    # 整数域 LUT 卷积 + 末端反量化）。量化全部 static 非对称：
    #   激活  per-tensor：q_x = clamp(round(x/s_x) + z_x, 0, 255)
    #   权重  per-channel：q_w = clamp(round(w/s_w[o]) + z_w[o], 0, 255)
    # scale/zero_point 由 min/max 统计 buffer 导出，训练时 EMA 更新
    # （update_scale 可冻结）；LUT 两个操作数都是原始 uint8 值 lut[q_x][q_w]。
    #
    # 反量化含零点交叉项（K = C·kH·kW 个 tap，对每个输出 (n,o,l)）：
    #   Σ_k (q_x−z_x)(q_w−z_w[o])
    #     = LUTSUM − z_w[o]·Σ_k q_x − z_x·Σ_k q_w + K·z_x·z_w[o]
    # LUTSUM 用近似乘法器（bgemm kernel）；修正项按精确算术在外面补——
    # 对应硬件上精确的行/列求和累加器（gemmlowp 同款做法）。其中
    # Σ_k q_w 是 per-channel 常数，Σ_k q_x 随输出位置变化（就是 unfold
    # 列和）。padding 注入 z_x（真实 0 的量化像），使修正式对含 padding
    # 的位置同样成立（padded tap 的 (q_x−z_x) 恰为 0）。
    #
    # 修正项全部用可微 torch op 写出，于是 bgemm 的 STE backward 只需给
    # 原始乘积的梯度（见 bgemm_uint8.py），autograd 自动合成
    # dy/dq_x = s·(q_w−z_w)、dy/dq_w = s·(q_x−z_x)。
    #
    # 目前仅 grad='ste'、groups=1；im2col 走 torch unfold（fp32），
    # conv 级 u8 融合路径与 lre 等后续再加。
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 lut: torch.Tensor,
                 grad: str = 'ste',
                 bias: torch.Tensor | bool | None = None,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 update_scale: bool = True,
                 scale_momentum: float = 0.05,
         ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        if groups != 1:
            raise NotImplementedError(
                f"Conv2d_uint8 only supports groups=1 for now, got groups={groups}")
        self.groups = groups
        if grad != 'ste':
            raise NotImplementedError(
                f"Conv2d_uint8 only supports grad='ste' for now, got grad='{grad}'")
        self.grad = grad
        self.qmin = 0
        self.qmax = 255
        self.scale_momentum = scale_momentum
        self.update_scale = update_scale

        # lut
        self.register_buffer('lut', lut)
        # weight
        self.weight = nn.Parameter(torch.empty(self.out_channels,
                                               self.in_channels,
                                               self.kernel_size[0], self.kernel_size[1]))

        # min/max 统计 buffer（scale/zero_point 每次 forward 由它们导出）。
        # 激活范围冷启动给 [0, 1]，前几步 EMA 会迅速贴合真实分布；
        # 校准 checkpoint 可直接 load_state_dict 覆盖。
        self.register_buffer('x_min', torch.tensor(0.0))
        self.register_buffer('x_max', torch.tensor(1.0))
        self.register_buffer('w_min', torch.zeros(self.out_channels))
        self.register_buffer('w_max', torch.ones(self.out_channels))

        # bias
        if isinstance(bias, torch.Tensor) or bias == True:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")

        self.reset_parameters()

        # 权重范围用当前权重的 per-channel min/max 初始化
        with torch.no_grad():
            reduce_dims = tuple(range(1, self.weight.dim()))
            self.w_min.copy_(self.weight.detach().amin(dim=reduce_dims))
            self.w_max.copy_(self.weight.detach().amax(dim=reduce_dims))

        # 外部给定 bias 时覆盖默认初始化
        if isinstance(bias, torch.Tensor):
            if bias.shape != (self.out_channels,):
                raise ValueError(f"bias must have shape ({self.out_channels},), got {tuple(bias.shape)}")
            with torch.no_grad():
                self.bias.copy_(bias)

    def reset_parameters(self):
        # 与 nn.Conv2d 相同的默认初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"Conv2d_uint8(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"grad={self.grad})"

    def unfreeze_scale(self):
        self.update_scale = True
    def freeze_scale(self):
        self.update_scale = False

    def _update_scale(self, x):
        # 激活 min/max 的 EMA。多卡 DDP 下先做全局 MIN/MAX 归约，
        # 保证每个 rank 的统计量（进而 scale/zero_point）完全一致。
        with torch.no_grad():
            mn = x.amin()
            mx = x.amax()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(mn, op=dist.ReduceOp.MIN)
                dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            m = self.scale_momentum
            self.x_min.copy_(m * mn + (1 - m) * self.x_min)
            self.x_max.copy_(m * mx + (1 - m) * self.x_max)

    def _update_scale_w(self):
        # 权重 per-channel min/max 的 EMA。DDP 下权重各 rank 逐位一致，
        # 统计量自然一致，无需归约。
        with torch.no_grad():
            reduce_dims = tuple(range(1, self.weight.dim()))
            mn = self.weight.detach().amin(dim=reduce_dims)   # [O]
            mx = self.weight.detach().amax(dim=reduce_dims)   # [O]
            m = self.scale_momentum
            self.w_min.copy_(m * mn + (1 - m) * self.w_min)
            self.w_max.copy_(m * mx + (1 - m) * self.w_max)

    def forward(self, x: torch.Tensor):

        # 0. output shape
        B, C, H, W = x.shape
        O, _, kH, kW = self.weight.shape
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # 1. quantization（static：scale/zero 由统计 buffer 导出）
        if self.training and self.update_scale:
            self._update_scale(x)
            self._update_scale_w()

        scale_x, zero_x = uint8_qparams(self.x_min, self.x_max, self.qmax)
        scale_w, zero_w = uint8_qparams(self.w_min, self.w_max, self.qmax)   # [O]

        xq = quantization.static_quantize_uint8(x, scale_x, zero_x,
                                                self.qmin, self.qmax)
        wq = quantization.static_quantize_uint8(self.weight, scale_w, zero_w,
                                                self.qmin, self.qmax, ch_axis=0)

        # 2. padding + im2col。padding 必须注入 z_x（真实 0 的量化像）而不是 0，
        # 否则零点修正式在边界位置不成立。之后 unfold 一律 padding=0。
        if pH or pW:
            xp = zero_x.view(1, 1, 1, 1).expand(B, C, H + 2 * pH, W + 2 * pW).clone()
            xp[:, :, pH:pH + H, pW:pW + W] = xq
            xq = xp
        if (kH, kW) == (1, 1):
            # 1x1：padding 已提前完成，unfold 只是 gather 复制，直接切片展平
            if self.stride != (1, 1):
                xq = xq[:, :, ::sH, ::sW]
            xu = xq.flatten(2)                                   # (B, C, L)
        else:
            xu = F.unfold(xq, self.kernel_size, dilation=self.dilation,
                          padding=0, stride=self.stride)         # (B, K, L)
        wu = wq.view(O, -1)                                      # (O, K)

        # 3. LUT-BGEMM（原始 uint8 值的近似乘积和）
        y = bgemm_uint8.bgemm_uint8_ste(xu, wu, self.lut)        # (B, O, L)

        # 4. 零点修正（可微：xsum/wsum 的梯度给 STE 反传补上 -z 项）
        #    Σ(q_x−z_x)(q_w−z_w) = LUTSUM − z_w·Σq_x − z_x·Σq_w + K·z_x·z_w
        Kdim = wu.shape[1]
        xsum = xu.sum(dim=1)                                     # (B, L)
        wsum = wu.sum(dim=1)                                     # (O,)
        zw = zero_w.view(1, -1, 1)
        y = y - zw * xsum.unsqueeze(1) - (zero_x * wsum).view(1, -1, 1) \
              + Kdim * zero_x * zw

        # 5. reshape + 反量化 + bias（与 int8 相同的 fused addcmul）
        y = y.view(B, O, OH, OW)
        s = (scale_x * scale_w).view(1, -1, 1, 1)
        if self.bias is not None:
            y = torch.addcmul(self.bias.view(1, -1, 1, 1), y, s)
        else:
            y = y * s

        return y
