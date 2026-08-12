"""静态 fake-quant（decoupled 风格：返回整数域 q，反量化由调用方完成）。

两种方案 × 两种粒度，qmin/qmax 上下界均可自定义：

  int8 对称（无 zero_point）      q = clamp(round(x/s),     qmin, qmax)
      反量化 x ≈ q * s              默认 [qmin, qmax] = [-127, 127]
  uint8 非对称（带 zero_point）   q = clamp(round(x/s + z), qmin, qmax)
      反量化 x ≈ (q - z) * s        默认 [qmin, qmax] = [0, 255]

  per-tensor:  ch_axis=None，scale / zero_point 为标量（1 元素 tensor 或 python 数）
  per-channel: ch_axis=int， scale / zero_point 形状 [x.shape[ch_axis]]

权重量化复用 static_quantize_int8 的 per-channel 路径，scale 由校准和 EMA 维护，
量化上下界由调用方按 3~8 bit 位宽传入。

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
