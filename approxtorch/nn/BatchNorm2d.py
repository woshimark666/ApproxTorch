import torch
import torch.nn as nn
import approxtorch.nn.FakeQuant as fq


class BatchNorm2dRequant(nn.Module):
    def __init__(self, num_features, scale_z, scale_momentum=0.1, **bn_kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **bn_kwargs)
        self.update_scale = True
        self.scale_momentum = scale_momentum  # EMA 动量，和 BN 默认值一致

        if not isinstance(scale_z, torch.Tensor):
            scale_z = torch.tensor(scale_z, dtype=torch.float32)
        self.register_buffer('scale_z', scale_z)


    def extra_repr(self) -> str:
        return (f"num_features={self.bn.num_features}, scale_z={self.scale_z.item():.6f}, "
                f"scale_momentum={self.scale_momentum}, update_scale={self.update_scale}, "
                f"bn_kwargs={self.bn.extra_repr()}")

    def freeze_scale(self):
        """冻结 scale，不再更新（用于 BatchNorm 融合后）"""
        self.update_scale = False
    
    def unfreeze_scale(self):
        """解冻 scale，继续更新（用于 BatchNorm 融合后）"""
        self.update_scale = True

    def _compute_current_scale(self, x):
        """从当前 batch 的 BN 输出计算 scale（per-tensor 对称量化）"""
        # int8 对称量化: scale = max(|x|) / 127
        return x.abs().max() / 127.0

    def forward(self, x):
        # 1. 标准 BN
        x = self.bn(x)

        # 2. EMA 更新 scale（仅训练时）
        if self.update_scale and self.training:
            with torch.no_grad():
                current_scale = self._compute_current_scale(x)
                # EMA: scale = (1 - momentum) * scale_old + momentum * scale_new
                self.scale_z.copy_(
                    (1 - self.scale_momentum) * self.scale_z + self.scale_momentum * current_scale
                )

        # 3. fake quantize (STE)
        x = fq.FakeQuantize_int8.apply(x, self.scale_z)
        return x