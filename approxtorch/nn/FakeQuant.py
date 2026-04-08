import torch
from torch.autograd import Function

class FakeQuantize_int8(Function):
    @staticmethod
    def forward(ctx, x, s):
        q_x = torch.clamp(torch.round(x / s ), -128, 127)
        hat_x = q_x * s
        return hat_x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

