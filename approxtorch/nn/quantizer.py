import torch
import torch.nn as nn
from torch.autograd import Function

def asymmetric_static_quantize_uint8_per_tensor(x, s, z, qmin, qmax):
    
    s = s.to(torch.float)
    z = z.to(torch.float)  
    
    u = x / s + z
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.uint8)


def asymmetric_static_quantize_uint8_per_channel(x, s, z, qmin, qmax):
    
    s = s.to(torch.float)
    z = z.to(torch.float)  
    
    s = s.view(-1,1,1,1)
    z = z.view(-1,1,1,1)
    
    u = x / s + z
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.uint8)

def symmetric_static_quantize_int8_per_tensor(x, s, z, qmin, qmax):
    s = s.to(torch.float)
    u = x / s 
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.int8)

def symmetric_static_quantize_int8_per_channel(x, s, z, qmin, qmax):
    s = s.view(-1,1,1,1)
    u = x / s 
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.int8)

def symmetric_dynamic_quantize_int8_per_tensor(x, qmin, qmax):
    absmax = torch.max(torch.abs(x))
    s = absmax / float((qmax - qmin)/2)
    q = torch.round(x / s).clamp(qmin, qmax)
    z = None
    return q.to(torch.int8), s, z

def symmetric_dynamic_quantize_int8_per_channel(x, qmin, qmax):
    absmax = torch.amax(torch.abs(x), dim=(1,2,3), keepdim=True)
    s = absmax / float((qmax - qmin)/2)
    s = s.detach()
    q = torch.round(x / s).clamp(qmin, qmax)
    z = None    
    return q.to(torch.int8), s.view(-1), z



# static asymmetric quantization for int8 per tensor
class _symmetric_static_quantize_int8_per_tensor(Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                s: torch.Tensor, 
                z: torch.Tensor,
                qmin: int, 
                qmax: int
                ):


        s = s.to(torch.float)
        z = z.to(torch.float)  
        
        u = x / s + z
        q = torch.round(u).clamp(qmin, qmax)

        ctx.save_for_backward(u, s)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return q.to(torch.uint8)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        u, s = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        gate = (u >= qmin) & (u <= qmax)
        gate = gate.to(dtype=grad_out.dtype)

        grad_x = grad_out * gate / s

        # only return grad for x
        return grad_x, None, None, None, None, None


class _symmetric_static_quantize_int8_pre_tensor(Function):

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
    return _symmetric_static_quantize_int8_pre_tensor.apply(x, s, qmin, qmax)

# to dynamicly quantize weights
def symmetric_dynamic_quantize_int8_per_channel(x, s, z, ch_axis=1, qmin=-127, qmax=127):
    """
    Symmetric dynamic per-channel int8 quantization.

    x: input tensor or weight tensor
       activation example: [N, C, H, W], ch_axis=1
       weight example:     [O, I, KH, KW], ch_axis=0

    s: optional scale placeholder, not used here
    z: optional zero_point placeholder, not used for symmetric quantization

    return:
        q:     quantized int8-like tensor, float dtype but integer values
        scale: per-channel scale, shape [C]
        z:     zero point, symmetric quantization uses 0
    """
    # reduce all dims except channel axis
    reduce_dims = [i for i in range(x.dim()) if i != ch_axis]

    # per-channel absmax
    # example:
    # x [N, C, H, W], ch_axis=1 -> absmax shape [C]
    # w [O, I, KH, KW], ch_axis=0 -> absmax shape [O]
    absmax = x.detach().abs().amax(dim=reduce_dims)

    # scale = max_abs / qmax
    scale = absmax / qmax

    # call previous per-channel quantizer
    q = _symmetric_static_quantize_int8_per_channel.apply(x, scale, ch_axis, qmin, qmax)

    return q


