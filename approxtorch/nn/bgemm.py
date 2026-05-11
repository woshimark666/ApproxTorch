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

        return at.backend.ops.bgemm_fake_int8_gpt(x, w, lut)



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
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        w = w.transpose(0, 1).contiguous() 
        dx = ctx.dx
        dw = ctx.dw
        grad_x, grad_w = at.backend.ops.bgemm_lre_backward(grad_output = grad_outputs,
                                                   x = x,
                                                   w = w,
                                                   dx = dx,
                                                   dw = dw)
        grad_w = grad_w.transpose(0, 1).contiguous()
        return grad_x, grad_w, None, None, None, None
    
def bgemm_int8_lre(x, w, lut, dx, dw):
    return _bgemm_int8_lre.apply(x, w, lut, dx, dw, None)


class _bgemm_int8_bqsg64(_bgemm_int8_base):

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        coeff = ctx.coeff

        grad_x, grad_w = at.backend.ops.bgemm_bqsg64_backward(grad_outputs, x, w, coeff)

        return grad_x, grad_w, None, None, None, None

def bgemm_int8_bqsg64(x, w, lut, coeff):
    return _bgemm_int8_bqsg64.apply(x, w, lut, None, None, coeff)