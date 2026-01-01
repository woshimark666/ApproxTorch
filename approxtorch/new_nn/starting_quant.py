import torch
import torch.nn as nn

class StartingQuant_int8(nn.Module):
    def __init__(self, scale):
        super().__init__()
        if scale is not None:
            self.scale = torch.nn.Parameter(scale, requires_grad=False)
        else:
            self.scale = torch.nn.Parameter(torch.empty([]), requires_grad=False)

    def forward(self, x):
        x = torch.round(x / self.scale)
        x = torch.clamp(x, -128, 127)
        x = x * self.scale
        return x