import torch
import torch.nn as nn

class FakeQuantCat(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale  # 这是从 model_int8 中拿到的 cat 层 scale

    def forward(self, x1, x2):
        # 1. 物理拼接
        x = torch.cat([x1, x2], dim=1)
        
        # 2. 伪量化模拟 (Requantization)
        # 模拟数据被重新压入 cat 层指定的 S 刻度中
        x = torch.round(x / self.scale)
        x = torch.clamp(x, -128, 127)
        x = x * self.scale
        
        return x