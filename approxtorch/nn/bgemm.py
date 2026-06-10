import torch
import torch.nn as nn
from torch.autograd import Function
import approxtorch as at

class _bgemm_int8_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw, coeff):
        ctx.save_for_backward(x, w)
        ctx.dx = dx
        ctx.dw = dw
        ctx.coeff = coeff

        return at.backend.ops.bgemm_fake_int8_claude(x, w, lut)



class _bgemm_int8_ste(_bgemm_int8_base):
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None, None, None, None


def bgemm_int8_ste(x, w, lut):
    return _bgemm_int8_ste.apply(x, w, lut, None, None, None)



class _bgemm_int8_lre(_bgemm_int8_base):

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw, xq_pre, geom):
        # 主链 float 不变；backward 载荷只存量化后的小数据：
        # - 给了 xq_pre（unfold 前的 int8 小图 [B,C,H,W]）+ geom 时，存它，
        #   backward 用隐式 im2col 算子按需展开（展开形态永不落显存）
        # - 否则回退为保存 forward 内部现成的 uint8 unfold 映像（仍比 fp32 省 4 倍）
        # 两条路径的梯度与 fp32 保存逐位一致
        y, xq, wq = at.backend.ops.bgemm_fake_int8_claude_save(x, w, lut)
        if xq_pre is not None and geom is not None:
            ctx.save_for_backward(xq_pre, wq)
            ctx.geom = geom
        else:
            ctx.save_for_backward(xq, wq)
            ctx.geom = None
        ctx.dx = dx
        ctx.dw = dw

        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        w = w.transpose(0, 1).contiguous()
        dx = ctx.dx
        dw = ctx.dw
        if ctx.geom is not None:
            kernel_size, stride, padding, dilation = ctx.geom
            grad_x, grad_w = at.backend.ops.bgemm_lre_backward_claude_im2col(
                grad_outputs, x, w, dx, dw, kernel_size, stride, padding, dilation)
        else:
            grad_x, grad_w = at.backend.ops.bgemm_lre_backward_claude(grad_output = grad_outputs,
                                                       x = x,
                                                       w = w,
                                                       dx = dx,
                                                       dw = dw)
        grad_w = grad_w.transpose(0, 1).contiguous()
        return grad_x, grad_w, None, None, None, None, None

def bgemm_int8_lre(x, w, lut, dx, dw, xq_pre=None, geom=None):
    return _bgemm_int8_lre.apply(x, w, lut, dx, dw, xq_pre, geom)


class _bgemm_int8_bqsg64(_bgemm_int8_base):

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        coeff = ctx.coeff

        grad_x, grad_w = at.backend.ops.bgemm_bqsg64_backward(grad_outputs, x, w, coeff)

        return grad_x, grad_w, None, None, None, None

def bgemm_int8_bqsg64(x, w, lut, coeff):
    return _bgemm_int8_bqsg64.apply(x, w, lut, None, None, coeff)