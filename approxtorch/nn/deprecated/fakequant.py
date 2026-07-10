import torch
from torch.autograd import Function


class _symmetric_static_quantize_int8_per_tensor(Function):

    @staticmethod
    def forward(ctx, x, scale, qmin=-127, qmax=127):
        ctx.qmin = qmin
        ctx.qmax = qmax

        # CUDA: 融合算子，forward 一个 kernel 出 (q, bool mask)，
        # 只保存 1 字节的 mask 而不是 fp32 的 x/scale（省 4 倍激活显存），
        # 数值与下面的原始路径逐位一致
        if x.is_cuda and x.dtype == torch.float32 and scale.is_cuda:
            q, mask = torch.ops.approxtorch.fakequant_per_tensor_claude.default(
                x, scale, qmin, qmax)
            ctx.save_for_backward(mask, scale)
            ctx.fused = True
            return q

        ctx.fused = False
        scale = torch.clamp(scale, min=1e-12)

        x = x / scale
        ctx.save_for_backward(x, scale)

        x = torch.round(x)
        x = torch.clamp(x, qmin, qmax)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        qmin = ctx.qmin
        qmax = ctx.qmax

        if ctx.fused:
            mask, scale = ctx.saved_tensors
            # STE: gx = grad_output * mask / scale，单 kernel
            grad_x = torch.ops.approxtorch.fakequant_per_tensor_backward_claude.default(
                grad_output, mask, scale)
            return grad_x, None, None, None

        scaled_x, scale = ctx.saved_tensors

        mask = (scaled_x >= qmin) & (scaled_x <= qmax)

        # 因为 forward 返回的是 q = round(x / scale)
        # STE: d round(x/scale) / d x ≈ 1 / scale
        grad_x = grad_output * mask.to(grad_output.dtype) / scale

        # static quantization 下 scale 一般不学习，所以直接 None
        grad_scale = None

        return grad_x, grad_scale, None, None



class _symmetric_static_quantize_int8_per_channel(Function):

    @staticmethod
    def forward(ctx, x, scale, ch_axis=1, qmin=-127, qmax=127):
        """
        x:     input tensor, e.g. [N, C, H, W]
        scale: per-channel scale, shape [C]
        ch_axis: channel dimension, default 1 for NCHW
        """

        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.ch_axis = ch_axis

        scale = torch.clamp(scale, min=1e-12)

        # reshape scale for broadcasting
        # example: x [N, C, H, W], scale [C] -> [1, C, 1, 1]
        view_shape = [1] * x.dim()
        view_shape[ch_axis] = -1
        scale_view = scale.view(*view_shape)

        scaled_x = x / scale_view

        ctx.save_for_backward(scaled_x, scale_view)

        q = torch.round(scaled_x)
        q = torch.clamp(q, qmin, qmax)

        return q

    @staticmethod
    def backward(ctx, grad_output):
        scaled_x, scale_view = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        mask = (scaled_x >= qmin) & (scaled_x <= qmax)

        # forward: q = round(x / scale)
        # STE: dq/dx ≈ 1 / scale
        grad_x = grad_output * mask.to(grad_output.dtype) / scale_view

        # static quantization: scale 不通过 autograd 学习
        grad_scale = None

        # 对应 forward 的参数:
        # x, scale, ch_axis, qmin, qmax
        return grad_x, grad_scale, None, None, None




def symmetric_static_quantize_int8_per_tensor(x, s, z, qmin=-127, qmax=127):
    return _symmetric_static_quantize_int8_per_tensor.apply(x, s, qmin, qmax)

# to dynamicly quantize weights
def symmetric_dynamic_quantize_int8_per_channel(x, ch_axis=1, bits=8, trunc_bits=0):
    """
    Symmetric dynamic per-channel signed quantization, supporting 3-bit to 8-bit.

    x:       input tensor or weight tensor
             activation example: [N, C, H, W], ch_axis=1
             weight example:     [O, I, KH, KW], ch_axis=0
    ch_axis: channel dimension
    bits:    bit-width of signed quantization (3~8), default 8
             qmax = 2^(bits-1) - 1,  qmin = -qmax  (symmetric, no -128)
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
        q:     quantized tensor, float dtype but integer values
               （trunc_bits>0 时取值恒为 B=2^trunc_bits 的倍数）
        scale: per-channel scale s_w^eff, shape [C]
    """
    assert 3 <= bits <= 8, f"bits must be between 3 and 8, got {bits}"
    assert trunc_bits >= 0, f"trunc_bits must be >= 0, got {trunc_bits}"

    qmax = 2 ** (bits - 1) - 1

    B = 1 << trunc_bits                 # 2^n，权重操作数的有效台阶间距
    K = qmax // B                       # floor(qmax / B)，单侧有效台阶数
    assert K >= 1, (
        f"trunc_bits={trunc_bits} too large for bits={bits}: "
        f"B=2^{trunc_bits}={B} > qmax={qmax}, no effective level left")

    # reduce all dims except channel axis
    # x [N, C, H, W], ch_axis=1 -> absmax shape [C]
    # w [O, I, KH, KW], ch_axis=0 -> absmax shape [O]
    reduce_dims = [i for i in range(x.dim()) if i != ch_axis]
    absmax = x.detach().abs().amax(dim=reduce_dims)

    # 有效 dequant scale：让最大权重正好映到 Qeff，而不是先映到 qmax 再被硬件截到
    # Qeff —— 否则最大权重会被重建成 (Qeff/qmax)*absmax，平白缩水动态范围。
    Qeff = B * K
    scale = absmax / Qeff               # s_w^eff（返回给上层做反量化）

    # 复用逐通道 STE 量化把 w 量化到台阶 index k ∈ [-K, K]（等价于 scale=absmax/K
    # 的普通 int8 量化，round / clip / STE-mask 全部沿用）。
    scale_inner = scale * B             # = absmax / K
    k = _symmetric_static_quantize_int8_per_channel.apply(x, scale_inner, ch_axis, -K, K)

    # 乘回 B 得到 B 的倍数权重整数。B 是常量标量，autograd 自动把 STE 梯度 ×B，于是
    # dq/dw ≈ B / scale_inner = 1 / scale = 1 / s_w^eff，正是正确的直通梯度。
    q = k * B

    return q, scale


def symmetric_static_quantize_int8_per_channel_grid(x, scale, ch_axis=0, bits=8, trunc_bits=0):
    """
    静态版 effective-grid 量化：scale 由外部给定（校准初始化 + 训练中 EMA），
    不再由当步 absmax 决定 —— 解决动态 absmax × 粗格点的阈值漂移/自指反馈问题。

    scale 语义 = 「传统」dequant scale（absmax/qmax 风格，即 calib.py 存的 scale_w），
    这样校准 checkpoint 可以直接加载。内部换算成有效格点 scale：

        qmax  = 2^(bits-1) - 1
        B     = 2^trunc_bits,  K = floor(qmax / B),  Qeff = B*K
        s_eff = scale * qmax / Qeff        # trunc_bits=0 时 Qeff=qmax，s_eff=scale
        k     = clip(round(x / (s_eff*B)), -K, K)
        q     = B * k                      # T_n(q) = q，硬件截断为 no-op

    落在 [-Qeff*s_eff, Qeff*s_eff] 之外的权重被 clip，STE 反传时梯度被 mask 置零
    （与静态激活量化同一行为）。

    return:
        q:     quantized tensor（B 的倍数整数，float dtype）
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

    s_eff = scale * (qmax / (B * K))
    scale_inner = s_eff * B
    k = _symmetric_static_quantize_int8_per_channel.apply(x, scale_inner, ch_axis, -K, K)
    q = k * B

    return q, s_eff