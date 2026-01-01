# Conv2d_int8_STE Scale 管理功能指南

## 概述

本指南介绍如何在 `Conv2d_int8_STE` 中实现和使用 scale 参数的更新和冻结功能。这些功能允许您灵活地控制量化过程中 scale 参数的行为。

## 功能特性

### 1. 支持的量化方法

- **Dynamic量化**: `('dynamic', 'tensor', 'tensor')` 或 `('dynamic', 'tensor', 'channel')`
- **Static量化**: `('static', 'tensor', 'tensor')` 或 `('static', 'tensor', 'channel')`
- **Trainable量化**: `('trainable', 'tensor', 'tensor')` 或 `('trainable', 'tensor', 'channel')`

### 2. Scale 管理功能

- **自动初始化**: 根据量化方法自动创建相应的 scale 参数
- **冻结/解冻**: 动态控制 scale 参数是否参与梯度更新
- **参数访问**: 提供便捷的方法获取当前的 scale 参数

## 实现细节

### 核心方法

#### 1. `_init_trainable_scales()`
```python
def _init_trainable_scales(self):
    """初始化可训练的scale参数"""
    if self.qmethod[0] == 'trainable':
        # 为feature创建可训练的scale (per tensor)
        self.scale_feature = torch.nn.Parameter(torch.ones(1))
        
        # 为weight创建可训练的scale
        if self.qmethod[2] == 'tensor':
            self.scale_weight = torch.nn.Parameter(torch.ones(1))  # per tensor
        elif self.qmethod[2] == 'channel':
            self.scale_weight = torch.nn.Parameter(torch.ones(self.out_channels))  # per channel
```

#### 2. `freeze_scales()`
```python
def freeze_scales(self):
    """冻结scale参数，使其不参与梯度更新"""
    self.freeze_scales = True
    if self.scale_feature is not None:
        self.scale_feature.requires_grad_(False)
    if self.scale_weight is not None:
        self.scale_weight.requires_grad_(False)
```

#### 3. `unfreeze_scales()`
```python
def unfreeze_scales(self):
    """解冻scale参数，使其参与梯度更新"""
    self.freeze_scales = False
    if self.scale_feature is not None:
        self.scale_feature.requires_grad_(True)
    if self.scale_weight is not None:
        self.scale_weight.requires_grad_(True)
```

#### 4. `get_scale_params()`
```python
def get_scale_params(self):
    """获取当前的scale参数，用于传递给底层函数"""
    if self.qmethod[0] == 'trainable':
        return self.scale_feature, self.scale_weight
    elif self.qmethod[0] == 'static':
        return self.static_scale_feature, self.static_scale_weight
    else:
        return None, None
```

## 使用示例

### 1. 基本使用

```python
import torch
from approxtorch.nn import Conv2d_int8_STE

# 创建LUT
lut = torch.randint(-127, 128, (256*256, 2), dtype=torch.int8)

# 创建可训练量化的卷积层
conv = Conv2d_int8_STE(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    lut=lut,
    qmethod=('trainable', 'tensor', 'tensor'),  # 可训练量化
    bias=True,
    padding=1,
    freeze_scales=False  # 初始时不冻结
)
```

### 2. 训练过程中管理 Scale

```python
import torch.optim as optim

# 创建优化器
optimizer = optim.Adam(conv.parameters(), lr=0.001)

# 第一阶段：训练scale参数
for epoch in range(10):
    optimizer.zero_grad()
    output = conv(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 冻结scale参数
conv.freeze_scales()

# 第二阶段：只训练权重
for epoch in range(5):
    optimizer.zero_grad()
    output = conv(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 解冻scale参数
conv.unfreeze_scales()

# 第三阶段：重新训练scale参数
for epoch in range(5):
    optimizer.zero_grad()
    output = conv(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### 3. 不同量化方法对比

```python
# Dynamic量化
conv_dynamic = Conv2d_int8_STE(
    in_channels=3, out_channels=16, kernel_size=3,
    lut=lut, qmethod=('dynamic', 'tensor', 'tensor'),
    bias=True, padding=1
)

# Static量化
static_scales = (torch.tensor([0.1]), torch.tensor([0.05]))
conv_static = Conv2d_int8_STE(
    in_channels=3, out_channels=16, kernel_size=3,
    lut=lut, qmethod=('static', 'tensor', 'tensor'),
    qparams=static_scales, bias=True, padding=1
)

# Trainable量化 (per-tensor)
conv_trainable = Conv2d_int8_STE(
    in_channels=3, out_channels=16, kernel_size=3,
    lut=lut, qmethod=('trainable', 'tensor', 'tensor'),
    bias=True, padding=1
)

# Trainable量化 (per-channel)
conv_trainable_channel = Conv2d_int8_STE(
    in_channels=3, out_channels=16, kernel_size=3,
    lut=lut, qmethod=('trainable', 'tensor', 'channel'),
    bias=True, padding=1
)
```

## 训练策略建议

### 1. 分阶段训练

1. **初始化阶段**: 使用 dynamic 量化或预训练的 static 量化参数
2. **联合训练阶段**: 同时训练权重和 scale 参数
3. **微调阶段**: 冻结 scale 参数，只训练权重
4. **最终优化阶段**: 解冻 scale 参数进行最终优化

### 2. 学习率设置

```python
# 为不同参数设置不同的学习率
optimizer = optim.Adam([
    {'params': conv.weight, 'lr': 0.001},
    {'params': conv.scale_feature, 'lr': 0.0001},  # scale参数使用较小的学习率
    {'params': conv.scale_weight, 'lr': 0.0001}
])
```

### 3. 监控 Scale 参数

```python
# 监控scale参数的变化
print(f"Scale feature: {conv.scale_feature.data.item():.6f}")
print(f"Scale weight: {conv.scale_weight.data.item():.6f}")
print(f"Scale feature gradient: {conv.scale_feature.grad}")
print(f"Scale weight gradient: {conv.scale_weight.grad}")
```

## 注意事项

1. **梯度检查**: 在冻结/解冻 scale 参数后，检查梯度是否正确传播
2. **参数保存**: 在保存模型时，确保包含 scale 参数的状态
3. **兼容性**: 确保底层 `conv2d_int8.py` 中的 trainable 量化方法已启用
4. **初始化**: 可训练 scale 参数初始化为 1.0，可以根据需要调整

## 故障排除

### 常见问题

1. **Scale 参数不更新**
   - 检查 `requires_grad` 属性
   - 确认优化器包含 scale 参数

2. **梯度为 None**
   - 检查量化方法是否正确
   - 确认底层函数支持相应的量化方法

3. **参数形状不匹配**
   - 检查 per-tensor 和 per-channel 的设置
   - 确认 out_channels 参数正确

### 调试技巧

```python
# 检查参数状态
print(f"Scale feature requires_grad: {conv.scale_feature.requires_grad}")
print(f"Scale weight requires_grad: {conv.scale_weight.requires_grad}")
print(f"Freeze scales flag: {conv.freeze_scales}")

# 检查梯度
if conv.scale_feature.grad is not None:
    print(f"Scale feature gradient norm: {conv.scale_feature.grad.norm()}")
if conv.scale_weight.grad is not None:
    print(f"Scale weight gradient norm: {conv.scale_weight.grad.norm()}")
```

## 总结

通过实现 scale 的更新和冻结功能，您可以：

1. **灵活控制训练过程**: 根据需要冻结或解冻 scale 参数
2. **优化量化精度**: 通过可训练 scale 参数提高量化效果
3. **实现分阶段训练**: 采用不同的训练策略优化模型性能
4. **监控训练状态**: 实时监控 scale 参数的变化和梯度

这些功能为量化神经网络的训练提供了更大的灵活性和控制能力。 