import torch
import torch.nn as nn
import torch.optim as optim
from approxtorch.nn.conv2d_int8 import conv2d_int8_STE
import approxtorch.approx_gemm as ap

class TrainableQuantizedConv2d(nn.Module):
    """
    可训练量化的Conv2d层
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 qmethod=('trainable', 'tensor', 'tensor')):
        super().__init__()
        
        # 创建权重参数
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # 卷积参数
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.qmethod = qmethod
        
        # 创建可训练的量化scale参数
        if qmethod[0] == 'trainable':
            # 为feature和weight创建可训练的scale
            self.scale_feature = nn.Parameter(torch.ones(1))  # per tensor
            
            if qmethod[2] == 'tensor':
                self.scale_weight = nn.Parameter(torch.ones(1))  # per tensor
            elif qmethod[2] == 'channel':
                self.scale_weight = nn.Parameter(torch.ones(out_channels))  # per channel
            else:
                raise ValueError(f"Unsupported quantization method: {qmethod}")
        else:
            self.scale_feature = None
            self.scale_weight = None
        
        # 创建LUT (这里用简单的随机初始化，实际应用中需要根据具体需求设置)
        self.lut = torch.randint(-127, 128, (256*256, 2), dtype=torch.int8)
        
    def forward(self, x):
        # 使用可训练量化的conv2d
        return conv2d_int8_STE(
            x, 
            self.weight, 
            self.lut,
            qmethod=self.qmethod,
            scale_feature=self.scale_feature,
            scale_weight=self.scale_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

# 示例使用
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建可训练量化的卷积层
    conv_layer = TrainableQuantizedConv2d(
        in_channels=3, 
        out_channels=16, 
        kernel_size=3, 
        padding=1,
        qmethod=('trainable', 'tensor', 'tensor')
    ).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(conv_layer.parameters(), lr=0.001)
    
    # 创建输入数据
    x = torch.randn(4, 3, 32, 32).to(device)
    target = torch.randn(4, 16, 32, 32).to(device)
    
    # 前向传播
    output = conv_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Scale feature: {conv_layer.scale_feature.data}")
    print(f"Scale weight: {conv_layer.scale_weight.data}")
    
    # 计算损失
    loss = nn.MSELoss()(output, target)
    print(f"Loss: {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"Scale feature gradient: {conv_layer.scale_feature.grad}")
    print(f"Scale weight gradient: {conv_layer.scale_weight.grad}")
    print(f"Weight gradient shape: {conv_layer.weight.grad.shape}")
    
    # 更新参数
    optimizer.step()
    
    print("Training step completed!")
    print(f"Updated scale feature: {conv_layer.scale_feature.data}")
    print(f"Updated scale weight: {conv_layer.scale_weight.data}") 