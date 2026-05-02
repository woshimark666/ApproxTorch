import torch
import torch.nn as nn
from torch.autograd import Function
import approxtorch as at

class bgemm_int8_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut):
        ctx.save_for_backward(x, w)
        return at.backend.ops.bgemm_fake_int8_gpt(x, w, lut)





class bgemm_int8_ste(Function):

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None