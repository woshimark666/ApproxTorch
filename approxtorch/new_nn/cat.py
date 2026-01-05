import torch
import torch.nn as nn
from .quantizer import LSQQuantizerFunc


class FakeQuantCat_int8(nn.Module):
    def __init__(self, scale):
        super().__init__()
        if scale is not None:
            self.scale = torch.nn.Parameter(scale, requires_grad=False)
        else:
            self.scale = torch.nn.Parameter(torch.empty([]), requires_grad=False)

    def forward(self, x1, x2):
        # 1. 物理拼接
        x = torch.cat([x1, x2], dim=1)
        
        # 2. 伪量化模拟 (Requantization)
        # 模拟数据被重新压入 cat 层指定的 S 刻度中
        x = torch.round(x / self.scale)
        x = torch.clamp(x, -128, 127)
        x = x * self.scale
        
        return x
    
    
class LSQCat_int8(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q_min, self.Q_max = -128, 127
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tensors, dim: int = 1):
        x = torch.cat(tensors, dim=dim)
        g = 1.0 / (x.numel() * self.Q_max) ** 0.5
        return LSQQuantizerFunc.apply(x, self.scale, self.Q_min, self.Q_max, g)

    def get_scale(self):
        return self.scale.item()