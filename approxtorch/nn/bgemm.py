import torch
import torch.nn as nn
from torch.autograd import Function
import approxtorch as at

class _bgemm_int8_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw):
        ctx.save_for_backward(x, w)
        ctx.dx = dx
        ctx.dw = dw

        return at.backend.ops.bgemm_fake_int8_gpt(x, w, lut)


class _bgemm_int8_ste(_bgemm_int8_base):

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None, None, None


def bgemm_int8_ste(x, w, lut):
    return _bgemm_int8_ste.apply(x, w, lut)



class _bgemm_int8_lre(_bgemm_int8_base):

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        