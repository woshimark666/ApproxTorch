import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, groups=1, act=True):
        padding = (kernel_size - 1) // 2
        modules = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            modules.append(nn.ReLU6(inplace=True))
        super().__init__(*modules)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # 1x1 expand
            layers.append(ConvBNAct(inp, hidden_dim, 1, 1))
        # 3x3 depthwise
        layers.append(ConvBNAct(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim))
        # 1x1 project (no activation)
        layers.append(ConvBNAct(hidden_dim, oup, 1, 1, act=False))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10:
    - 第一层改为 3x3 stride=1，避免 32x32 输入过早下采样
    - 其余 stage 的 stride 保持 ImageNet 配置（少量下采样到 2x2/4x4，再全局池化）
    """
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # t(c expansion), c(out channels), n(repeats), s(stride)
        inverted_residual_setting = [
            # CIFAR: 第一组不下采样
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # stem: 3x3 s=1
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        features = [ConvBNAct(3, input_channel, 3, 1)]  # keep 32x32

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # last conv
        features.append(ConvBNAct(input_channel, self.last_channel, 1, 1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # CIFAR spatial sizes end up ~2x2; use global average pooling
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    model = MobileNetV2(num_classes=10)
    print(model)
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    print(output.shape)