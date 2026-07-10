"""静态 fake-quant（decoupled 风格：返回整数域 q，反量化由调用方完成）。

两种方案 × 两种粒度，qmin/qmax 上下界均可自定义：

  int8 对称（无 zero_point）      q = clamp(round(x/s),     qmin, qmax)
      反量化 x ≈ q * s              默认 [qmin, qmax] = [-127, 127]
  uint8 非对称（带 zero_point）   q = clamp(round(x/s + z), qmin, qmax)
      反量化 x ≈ (q - z) * s        默认 [qmin, qmax] = [0, 255]

  per-tensor:  ch_axis=None，scale / zero_point 为标量（1 元素 tensor 或 python 数）
  per-channel: ch_axis=int， scale / zero_point 形状 [x.shape[ch_axis]]

权重量化（3~8 bit、支持近似乘法器 effective-grid 截断 trunc_bits，均 per-channel）：

  dynamic_quantize_int8      scale 取自当步 absmax（动态）
  static_quantize_int8_grid  scale 外部给定（calib 校准 + EMA，静态格点）

STE 反传：gx = go * mask / s，mask = 1{qmin <= x/s (+z) <= qmax}（clip 区间外
梯度置零）；scale / zero_point 属静态量化参数（由校准 / EMA 维护），不走 autograd。

per-tensor 的 fp32 CUDA 输入走 backend 融合算子（单 kernel 同时输出 (q, mask)，
激活只保存 1 字节 mask，省 4 倍显存，见 csrc/claude/quantization.cu）；其余情况
（CPU / 其他 dtype / per-channel）为纯 PyTorch 实现，两条路径数值逐位一致。
"""

import torch
from torch.autograd import Function


# ---------------------------------------------------------------- per-tensor

def _fusable(x, *params):
    # backend 融合算子只接受 fp32 CUDA（scale / zero_point 为 1 元素 fp32 CUDA tensor）
    return (x.is_cuda and x.dtype == torch.float32
            and all(p.is_cuda and p.dtype == torch.float32 and p.numel() == 1
                    for p in params))


class _static_quantize_per_tensor_symmetric(Function):

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        if _fusable(x, scale):
            q, mask = torch.ops.approxtorch.fakequant_per_tensor_claude.default(
                x, scale, qmin, qmax)
            ctx.save_for_backward(mask, scale)  # 融合 backward 内部会 clamp scale
            ctx.fused = True
            return q
        ctx.fused = False
        s = torch.clamp(scale, min=1e-12)
        v = x / s
        mask = (v >= qmin) & (v <= qmax)
        ctx.save_for_backward(mask, s)
        return torch.clamp(torch.round(v), qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        mask, s = ctx.saved_tensors
        if ctx.fused:
            grad_x = torch.ops.approxtorch.fakequant_per_tensor_backward_claude.default(
                grad_output, mask, s)
        else:
            grad_x = grad_output * mask.to(grad_output.dtype) / s
        return grad_x, None, None, None


class _static_quantize_per_tensor_asymmetric(Function):

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        if _fusable(x, scale, zero_point):
            q, mask = torch.ops.approxtorch.fakequant_per_tensor_asymmetric_claude.default(
                x, scale, zero_point, qmin, qmax)
            ctx.save_for_backward(mask, scale)
            ctx.fused = True
            return q
        ctx.fused = False
        s = torch.clamp(scale, min=1e-12)
        v = x / s + zero_point
        mask = (v >= qmin) & (v <= qmax)
        ctx.save_for_backward(mask, s)
        return torch.clamp(torch.round(v), qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        # zero_point 是加性项，dq/dx = mask / s，与对称版完全一致
        mask, s = ctx.saved_tensors
        if ctx.fused:
            grad_x = torch.ops.approxtorch.fakequant_per_tensor_asymmetric_backward_claude.default(
                grad_output, mask, s)
        else:
            grad_x = grad_output * mask.to(grad_output.dtype) / s
        return grad_x, None, None, None, None


# --------------------------------------------------------------- per-channel
# 纯 PyTorch 路径（暂无融合 kernel）。与 per-tensor 相同的 STE 语义，
# backward 载荷同样只有 1 字节 mask + scale 视图。

class _static_quantize_per_channel_symmetric(Function):

    @staticmethod
    def forward(ctx, x, scale, ch_axis, qmin, qmax):
        view = [1] * x.dim()
        view[ch_axis] = -1
        s = torch.clamp(scale, min=1e-12).view(view)
        v = x / s
        mask = (v >= qmin) & (v <= qmax)
        ctx.save_for_backward(mask, s)
        return torch.clamp(torch.round(v), qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        mask, s = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype) / s, None, None, None, None


class _static_quantize_per_channel_asymmetric(Function):

    @staticmethod
    def forward(ctx, x, scale, zero_point, ch_axis, qmin, qmax):
        view = [1] * x.dim()
        view[ch_axis] = -1
        s = torch.clamp(scale, min=1e-12).view(view)
        z = zero_point.view(view)
        v = x / s + z
        mask = (v >= qmin) & (v <= qmax)
        ctx.save_for_backward(mask, s)
        return torch.clamp(torch.round(v), qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        # z 加性、无梯度贡献，backward 与对称 per-channel 一致
        mask, s = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype) / s, None, None, None, None, None


# ------------------------------------------------------------------ 公开 API

def _as_param_tensor(v, x):
    # 便利转换：python 数 -> 与 x 同设备的 fp32 标量 tensor
    # （热路径建议直接传注册好的 buffer，避免每步一次 H2D 拷贝）
    if isinstance(v, torch.Tensor):
        return v
    return torch.tensor(float(v), dtype=torch.float32, device=x.device)


def _check_per_channel_param(name, p, x, ch_axis):
    if p.numel() != x.shape[ch_axis]:
        raise ValueError(
            f"per-channel {name} must have {x.shape[ch_axis]} elements "
            f"(= x.shape[{ch_axis}]), got {p.numel()}")


def static_quantize_int8(x, scale, qmin=-127, qmax=127, ch_axis=None):
    """int8 静态对称量化：q = clamp(round(x/scale), qmin, qmax)，反量化 x ≈ q*scale。

    x:         输入 float tensor
    scale:     per-tensor: 标量（1 元素 tensor / python 数）
               per-channel: 形状 [x.shape[ch_axis]] 的 tensor
    qmin/qmax: 量化上下界，默认 [-127, 127]（对称、不含 -128），可自定义
               （如 6-bit 对称用 [-31, 31]）
    ch_axis:   None = per-tensor；int = per-channel 的通道维
               （激活 NCHW 用 1，权重 OIHW 用 0）
    return:    q —— float dtype 的整数值 tensor
    """
    if qmin >= qmax:
        raise ValueError(f"qmin must be < qmax, got [{qmin}, {qmax}]")
    scale = _as_param_tensor(scale, x)
    if ch_axis is None:
        return _static_quantize_per_tensor_symmetric.apply(x, scale, qmin, qmax)
    _check_per_channel_param('scale', scale, x, ch_axis)
    return _static_quantize_per_channel_symmetric.apply(x, scale, ch_axis, qmin, qmax)


def dynamic_quantize_int8(x, ch_axis=0, bits=8, trunc_bits=0):
    """对称动态 per-channel 量化（3~8 bit），scale 取自当步 per-channel absmax。
    主要用于权重（OIHW 用 ch_axis=0）。

    bits:    有符号量化位宽（3~8），qmax = 2^(bits-1) - 1，qmin = -qmax（对称、无 -128）
    trunc_bits: 近似乘法器对「权重操作数」截断的低位数 n（默认 0 = 普通逐点量化）。
             —— Effective-Grid Weight Quantization for Truncated-Operand ApproxConv ——

             近似乘法器形式  ApproxMul_n(q_x, q_w) = q_x * T_n(q_w)，
             其中  T_n(q_w) = 2^n * floor(q_w / 2^n)  把权重低 n 位截掉。这些低位在
             forward 里硬件根本看不见，与其把权重量化成任意整数再被硬件截一刀（round
             之后再 floor，带系统性偏置、还白白缩水动态范围），不如直接量化到硬件能
             分辨的「有效格点」—— B = 2^n 的倍数上，使 T_n(q_w^eff) = q_w^eff（截断
             退化成 no-op，网络也不再浪费优化去学那些无效低位）：

                 B    = 2^n
                 K    = floor(qmax / B)              # 单侧有效台阶数
                 Qeff = B * K                        # 正方向最大有效整数（<= qmax）
                 s    = absmax / Qeff                # 有效 dequant scale  s_w^eff
                 k    = clip(round(w / (s*B)), -K, K)
                 q    = B * k                        # 落在 {-KB,...,-B,0,B,...,KB}

             例 bits=8, n=4: B=16, K=7, Qeff=112，权重落在 {-112,-96,...,96,112}。
             n=0 时 B=1, K=qmax，与原始逐点量化逐位等价（完全向后兼容）。

             前提：被截断的必须是「权重」操作数。本仓库 forward 的 LUT 寻址为
             lut[(q_x+128)*256 + (q_w+128)]，权重在低字节、正是被截低 n 位的列；又因
             128 是 2^n 的倍数 (n<=7)，index 空间的 floor 截断与带符号值的 T_n 完全
             一致，故 q = B*k 恰好落在每个 2^n 台阶的代表点上。

    return:
        q:     量化结果（float dtype 的整数值；trunc_bits>0 时恒为 B=2^trunc_bits 的倍数）
        scale: per-channel 有效 dequant scale s_w^eff，shape [x.shape[ch_axis]]
    """
    assert 3 <= bits <= 8, f"bits must be between 3 and 8, got {bits}"
    assert trunc_bits >= 0, f"trunc_bits must be >= 0, got {trunc_bits}"

    qmax = 2 ** (bits - 1) - 1
    B = 1 << trunc_bits                 # 2^n，权重操作数的有效台阶间距
    K = qmax // B                       # floor(qmax / B)，单侧有效台阶数
    assert K >= 1, (
        f"trunc_bits={trunc_bits} too large for bits={bits}: "
        f"B=2^{trunc_bits}={B} > qmax={qmax}, no effective level left")

    reduce_dims = [i for i in range(x.dim()) if i != ch_axis]
    absmax = x.detach().abs().amax(dim=reduce_dims)

    # 有效 dequant scale：让最大权重正好映到 Qeff，而不是先映到 qmax 再被硬件截到
    # Qeff —— 否则最大权重会被重建成 (Qeff/qmax)*absmax，平白缩水动态范围。
    Qeff = B * K
    scale = absmax / Qeff               # s_w^eff（返回给上层做反量化）

    # 复用逐通道 STE 量化把 w 量化到台阶 index k ∈ [-K, K]（等价于 scale=absmax/K
    # 的普通 int8 量化，round / clip / STE-mask 全部沿用）。
    scale_inner = scale * B             # = absmax / K
    k = _static_quantize_per_channel_symmetric.apply(x, scale_inner, ch_axis, -K, K)

    # 乘回 B 得到 B 的倍数权重整数。B 是常量标量，autograd 自动把 STE 梯度 ×B，于是
    # dq/dw ≈ B / scale_inner = 1 / scale = 1 / s_w^eff，正是正确的直通梯度。
    q = k * B

    return q, scale


def static_quantize_int8_grid(x, scale, ch_axis=0, bits=8, trunc_bits=0):
    """静态版 effective-grid 量化：scale 由外部给定（校准初始化 + 训练中 EMA），
    不再由当步 absmax 决定 —— 解决动态 absmax × 粗格点的阈值漂移/自指反馈问题。

    scale 语义 = 「传统」dequant scale（absmax/qmax 风格，即 calib.py 存的 scale_w），
    这样校准 checkpoint 可以直接加载。内部换算成有效格点 scale：

        qmax  = 2^(bits-1) - 1
        B     = 2^trunc_bits,  K = floor(qmax / B),  Qeff = B*K
        s_eff = scale * qmax / Qeff        # trunc_bits=0 时 Qeff=qmax，s_eff=scale
        k     = clip(round(x / (s_eff*B)), -K, K)
        q     = B * k                      # T_n(q) = q，硬件截断为 no-op

    落在 [-Qeff*s_eff, Qeff*s_eff] 之外的权重被 clip，STE 反传时梯度被 mask 置零
    （与静态激活量化同一行为）。effective-grid 的动机与推导见 dynamic_quantize_int8。

    return:
        q:     量化结果（B 的倍数整数，float dtype）
        s_eff: 有效 dequant scale，shape 同 scale，调用方用它反量化
    """
    assert 3 <= bits <= 8, f"bits must be between 3 and 8, got {bits}"
    assert trunc_bits >= 0, f"trunc_bits must be >= 0, got {trunc_bits}"

    qmax = 2 ** (bits - 1) - 1
    B = 1 << trunc_bits
    K = qmax // B
    assert K >= 1, (
        f"trunc_bits={trunc_bits} too large for bits={bits}: "
        f"B=2^{trunc_bits}={B} > qmax={qmax}, no effective level left")

    _check_per_channel_param('scale', scale, x, ch_axis)
    s_eff = scale * (qmax / (B * K))
    scale_inner = s_eff * B
    k = _static_quantize_per_channel_symmetric.apply(x, scale_inner, ch_axis, -K, K)
    q = k * B

    return q, s_eff


def static_quantize_uint8(x, scale, zero_point, qmin=0, qmax=255, ch_axis=None):
    """uint8 静态非对称量化：q = clamp(round(x/scale + zero_point), qmin, qmax)，
    反量化 x ≈ (q - zero_point) * scale。

    zero_point: 约定为整数值（保证真实 0 恰好落在格点上）；per-tensor 标量 /
                per-channel [x.shape[ch_axis]]，与 scale 一样不走 autograd
    qmin/qmax:  默认 [0, 255]，可自定义（如 uint4 用 [0, 15]）
    其余参数、返回值同 static_quantize_int8
    """
    if qmin >= qmax:
        raise ValueError(f"qmin must be < qmax, got [{qmin}, {qmax}]")
    scale = _as_param_tensor(scale, x)
    zero_point = _as_param_tensor(zero_point, x)
    if ch_axis is None:
        return _static_quantize_per_tensor_asymmetric.apply(x, scale, zero_point, qmin, qmax)
    _check_per_channel_param('scale', scale, x, ch_axis)
    _check_per_channel_param('zero_point', zero_point, x, ch_axis)
    return _static_quantize_per_channel_asymmetric.apply(x, scale, zero_point, ch_axis, qmin, qmax)
