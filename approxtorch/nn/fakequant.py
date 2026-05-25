import torch
from torch.autograd import Function


class _symmetric_static_quantize_int8_per_tensor(Function):

    @staticmethod
    def forward(ctx, x, scale, qmin=-127, qmax=127):
        ctx.qmin = qmin
        ctx.qmax = qmax


        scale = torch.clamp(scale, min=1e-12)

        x = x / scale
        ctx.save_for_backward(x, scale)

        x = torch.round(x)
        x = torch.clamp(x, qmin, qmax)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        scaled_x, scale = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax
    
        mask = (scaled_x >= qmin) & (scaled_x <= qmax)

        # 因为 forward 返回的是 q = round(x / scale)
        # STE: d round(x/scale) / d x ≈ 1 / scale
        grad_x = grad_output * mask.to(grad_output.dtype) / scale

        # static quantization 下 scale 一般不学习，所以直接 None
        grad_scale = None

        return grad_x, grad_scale, None, None, None, None



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
def symmetric_dynamic_quantize_int8_per_channel(x, ch_axis=1, bits=8):
    """
    Symmetric dynamic per-channel signed quantization, supporting 3-bit to 8-bit.

    x:       input tensor or weight tensor
             activation example: [N, C, H, W], ch_axis=1
             weight example:     [O, I, KH, KW], ch_axis=0
    ch_axis: channel dimension
    bits:    bit-width of signed quantization (3~8), default 8
             qmax = 2^(bits-1) - 1,  qmin = -qmax  (symmetric, no -128)

    return:
        q:     quantized tensor, float dtype but integer values
        scale: per-channel scale, shape [C]
    """
    assert 3 <= bits <= 8, f"bits must be between 3 and 8, got {bits}"

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax

    # reduce all dims except channel axis
    reduce_dims = [i for i in range(x.dim()) if i != ch_axis]

    # per-channel absmax
    # x [N, C, H, W], ch_axis=1 -> absmax shape [C]
    # w [O, I, KH, KW], ch_axis=0 -> absmax shape [O]
    absmax = x.detach().abs().amax(dim=reduce_dims)

    scale = absmax / qmax

    q = _symmetric_static_quantize_int8_per_channel.apply(x, scale, ch_axis, qmin, qmax)

    return q, scale