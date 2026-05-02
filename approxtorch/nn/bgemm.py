import torch
import torch.nn as nn
from torch.autograd import Function


class _bgemm_forward_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut)


