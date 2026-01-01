import torch
import torch.nn as nn
import torch.optim as optim
from approxtorch.nn import Conv2d_int8_STE

class ScaleManagementExample(nn.Module):
    """
    展示scale更新和冻结功能的示例模型
    """
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3):
        super().__init__()
        
        # 创建LUT (查找表)
        self.lut = torch.randint(-127, 128, (256*256, 2), dtype=torch.int8)
        
        # 创建可训练量化的卷积层
        self.conv = Conv2d_int8_STE(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            lut=self.lut,
            qmethod=('trainable', 'tensor', 'tensor'),  # 可训练量化
            bias=True,
            padding=1,
            freeze_scales=False  # 初始时不冻结
        )
        
    def forward(self, x):
        return self.conv(x)

def demonstrate_scale_management():
    """演示scale的更新和冻结功能"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    model = ScaleManagementExample().to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建输入数据
    x = torch.randn(4, 3, 32, 32).to(device)
    target = torch.randn(4, 16, 32, 32).to(device)
    
    print("=== 初始状态 ===")
    print(f"Scale feature: {model.conv.scale_feature.data}")
    print(f"Scale weight: {model.conv.scale_weight.data}")
    print(f"Scale feature requires_grad: {model.conv.scale_feature.requires_grad}")
    print(f"Scale weight requires_grad: {model.conv.scale_weight.requires_grad}")
    
    # 第一阶段：训练scale参数
    print("\n=== 第一阶段：训练scale参数 ===")
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # 检查梯度
        if epoch == 0:
            print(f"Scale feature gradient: {model.conv.scale_feature.grad}")
            print(f"Scale weight gradient: {model.conv.scale_weight.grad}")
        
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            print(f"Scale feature: {model.conv.scale_feature.data.item():.6f}")
            print(f"Scale weight: {model.conv.scale_weight.data.item():.6f}")
    
    # 冻结scale参数
    print("\n=== 冻结scale参数 ===")
    model.conv.freeze_scales()
    print(f"Scale feature requires_grad: {model.conv.scale_feature.requires_grad}")
    print(f"Scale weight requires_grad: {model.conv.scale_weight.requires_grad}")
    print(f"Freeze scales flag: {model.conv.freeze_scales}")
    
    # 第二阶段：只训练权重，不训练scale
    print("\n=== 第二阶段：只训练权重（scale已冻结）===")
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # 检查梯度
        if epoch == 0:
            print(f"Scale feature gradient: {model.conv.scale_feature.grad}")
            print(f"Scale weight gradient: {model.conv.scale_weight.grad}")
            print(f"Weight gradient norm: {model.conv.weight.grad.norm()}")
        
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        print(f"Scale feature: {model.conv.scale_feature.data.item():.6f}")
        print(f"Scale weight: {model.conv.scale_weight.data.item():.6f}")
    
    # 解冻scale参数
    print("\n=== 解冻scale参数 ===")
    model.conv.unfreeze_scales()
    print(f"Scale feature requires_grad: {model.conv.scale_feature.requires_grad}")
    print(f"Scale weight requires_grad: {model.conv.scale_weight.requires_grad}")
    print(f"Freeze scales flag: {model.conv.freeze_scales}")
    
    # 第三阶段：重新训练scale参数
    print("\n=== 第三阶段：重新训练scale参数 ===")
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # 检查梯度
        if epoch == 0:
            print(f"Scale feature gradient: {model.conv.scale_feature.grad}")
            print(f"Scale weight gradient: {model.conv.scale_weight.grad}")
        
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        print(f"Scale feature: {model.conv.scale_feature.data.item():.6f}")
        print(f"Scale weight: {model.conv.scale_weight.data.item():.6f}")

def demonstrate_different_quantization_methods():
    """演示不同的量化方法"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut = torch.randint(-127, 128, (256*256, 2), dtype=torch.int8)
    
    print("\n=== 不同量化方法对比 ===")
    
    # 1. Dynamic量化
    conv_dynamic = Conv2d_int8_STE(
        in_channels=3, out_channels=16, kernel_size=3,
        lut=lut, qmethod=('dynamic', 'tensor', 'tensor'),
        bias=True, padding=1
    ).to(device)
    
    # 2. Static量化
    static_scales = (torch.tensor([0.1]), torch.tensor([0.05]))
    conv_static = Conv2d_int8_STE(
        in_channels=3, out_channels=16, kernel_size=3,
        lut=lut, qmethod=('static', 'tensor', 'tensor'),
        qparams=static_scales, bias=True, padding=1
    ).to(device)
    
    # 3. Trainable量化
    conv_trainable = Conv2d_int8_STE(
        in_channels=3, out_channels=16, kernel_size=3,
        lut=lut, qmethod=('trainable', 'tensor', 'tensor'),
        bias=True, padding=1
    ).to(device)
    
    # 4. Trainable per-channel量化
    conv_trainable_channel = Conv2d_int8_STE(
        in_channels=3, out_channels=16, kernel_size=3,
        lut=lut, qmethod=('trainable', 'tensor', 'channel'),
        bias=True, padding=1
    ).to(device)
    
    x = torch.randn(2, 3, 16, 16).to(device)
    
    print("Dynamic量化:")
    print(conv_dynamic)
    output_dynamic = conv_dynamic(x)
    print(f"Output shape: {output_dynamic.shape}")
    
    print("\nStatic量化:")
    print(conv_static)
    output_static = conv_static(x)
    print(f"Output shape: {output_static.shape}")
    
    print("\nTrainable量化 (per-tensor):")
    print(conv_trainable)
    output_trainable = conv_trainable(x)
    print(f"Output shape: {output_trainable.shape}")
    print(f"Scale feature: {conv_trainable.scale_feature.data}")
    print(f"Scale weight: {conv_trainable.scale_weight.data}")
    
    print("\nTrainable量化 (per-channel):")
    print(conv_trainable_channel)
    output_trainable_channel = conv_trainable_channel(x)
    print(f"Output shape: {output_trainable_channel.shape}")
    print(f"Scale feature: {conv_trainable_channel.scale_feature.data}")
    print(f"Scale weight: {conv_trainable_channel.scale_weight.data}")

if __name__ == "__main__":
    print("Scale管理功能演示")
    print("=" * 50)
    
    # 演示scale更新和冻结功能
    demonstrate_scale_management()
    
    # 演示不同量化方法
    demonstrate_different_quantization_methods()
    
    print("\n演示完成！") 