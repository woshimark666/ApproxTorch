import torch


# quantization from float to int8

def quantize_dynamic_int8(x: torch.Tensor, dim: tuple):  # scale is T/127
    """
        Quantize a float tensor to int8, clamped to [-127, 127]
        The scale is dynamically calculated based on the maximum absolute value of the tensor.
        symmetric quantization for signed int8
    """
    with torch.no_grad():
        abs_max = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        scale = abs_max / 127.
        x = torch.round(x / scale)
        x = torch.clamp(x, -127, 127)
        scale = scale.squeeze()
        return x, scale

def quantize_dynamic_int8_per_tensor(x: torch.Tensor):
    """
        this is for CNN per tensor dynamic quantization for int8 with 4D shape tensor
        acitivatio(N,C,H,W) or weight(O,C,H,W)
    """
    return quantize_dynamic_int8(x, (0,1,2,3))

def quantize_dynamic_int8_per_channel(x: torch.Tensor):
    """
        this is for CNN per channel dynamic quantization for int8 with 4D shape tensor
        mainly for weight(O,C,H,W)
    """
    return quantize_dynamic_int8(x, (1,2,3))


def quantize_static_int8_per_tensor(x: torch.Tensor, scale: torch.Tensor):
    """
        Quantize a float tensor to int8 with static quantization.
        Per tensor quantization 
        Symmetric quantization for signed int8
        Clampped to [-127, 127]
    """
    with torch.no_grad():
        x = torch.round(x / scale)
        x = torch.clamp(x, -127, 127)
        return x

def quantize_static_int8_per_channel(x: torch.Tensor, scale: torch.Tensor):
    """
        Quantize a float tensor to int8 with static quantization.
        Per channel quantization 
        Symmetric quantization for signed int8
        Clampped to [-127, 127]
    """
    with torch.no_grad():
        x = torch.round(x / scale.view(-1,1,1,1))
        x = torch.clamp(x, -127, 127)
        return x


def quantize_dynamic_uint8(x: torch.Tensor, dim: tuple):
    """
        Quantize a float tensor to uint8, clamped to [0,255]
        asymmetric quantization for uint8
    """
    with torch.no_grad():
        min_val = torch.amin(x, dim=dim, keepdim=True)
        max_val = torch.amax(x, dim=dim, keepdim=True)
        scale = (max_val - min_val) / 255.
        zero_point = - torch.round(min_val / scale)
        x = torch.round(x / scale + zero_point)
        x = torch.clamp(x, 0, 255)
        scale = scale.squeeze()
        zero_point = zero_point.squeeze()
        return x, scale, zero_point

def quantize_dynamic_uint8_per_tensor(x: torch.Tensor):
    """
        per tensor quantization for uint8
        asymmetric quantization
    """
    return quantize_dynamic_uint8(x, (0,1,2,3))

def quantize_dynamic_uint8_per_channel(x: torch.Tensor):
    """
        per channel quantization for uint8
        asymmetric quantization
    """
    return quantize_dynamic_uint8(x, (1,2,3))

def quantize_static_uint8_per_tensor(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """
        per tensor static quantization for uint8
    """
    with torch.no_grad():
        x = torch.round(x / scale + zero_point)
        x = torch.clamp(x, 0, 255)
        return x

def quantize_static_uint8_per_channel(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """
        per channel static quantization for uint8
    """
    with torch.no_grad():
        x = torch.round(x / scale.view(-1,1,1,1) + zero_point.view(-1,1,1,1))
        x = torch.clamp(x, 0, 255)
        return x

class TrainableQuantizeInt8PerTensor(torch.autograd.Function):
    """
    可训练的int8量化函数，使用STE (Straight-Through Estimator)
    支持scale参数的梯度反向传播
    """
    @staticmethod
    def forward(ctx, x, scale):
        # 前向传播：执行量化
        x_quantized = torch.round(x / scale)
        x_quantized = torch.clamp(x_quantized, -127, 127)
        return x_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：使用STE，梯度直接传递
        # grad_output对应量化后的梯度
        return grad_output, grad_output.sum()  # 返回对x和scale的梯度

class TrainableQuantizeInt8PerChannel(torch.autograd.Function):
    """
    可训练的int8逐通道量化函数，使用STE
    支持scale参数的梯度反向传播
    """
    @staticmethod
    def forward(ctx, x, scale):
        # x的形状通常是(O, C, H, W)，scale的形状是(O,)
        ctx.save_for_backward(x, scale)
        scale_reshaped = scale.view(-1, 1, 1, 1)
        x_quantized = torch.round(x / scale_reshaped)
        x_quantized = torch.clamp(x_quantized, -127, 127)
        return x_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # 对x的梯度：直接传递
        grad_x = grad_output
        
        # 对scale的梯度：需要考虑scale对量化的影响
        # d(round(x/scale))/d(scale) ≈ d(x/scale)/d(scale) = -x/scale^2
        # 但在STE中，我们通常简化为直接传递
        scale_reshaped = scale.view(-1, 1, 1, 1)
        grad_scale = (grad_output * (-x / (scale_reshaped ** 2))).sum(dim=(1, 2, 3))
        
        return grad_x, grad_scale

def quantize_trainable_int8_per_tensor(x: torch.Tensor, scale: torch.Tensor):
    """
    可训练的per tensor int8量化
    Args:
        x: 输入张量
        scale: 可训练的scale参数
    Returns:
        量化后的张量
    """
    return TrainableQuantizeInt8PerTensor.apply(x, scale)

def quantize_trainable_int8_per_channel(x: torch.Tensor, scale: torch.Tensor):
    """
    可训练的per channel int8量化
    Args:
        x: 输入张量 (O, C, H, W)
        scale: 可训练的scale参数 (O,)
    Returns:
        量化后的张量
    """
    return TrainableQuantizeInt8PerChannel.apply(x, scale)

if __name__ == "__main__":
    device = torch.device("cuda")
    activation = torch.randn((32,4,8,8), device=device)
    weight = torch.randn((16,4,3,3), device=device)
    
    # min_val = torch.amin(activation, dim=(0,1,2,3), keepdim=False)
    # max_val = torch.amax(activation, dim=(0,1,2,3), keepdim=False)
    # act_scale = (max_val - min_val) / 255.
    # zero_point = - torch.round(min_val / act_scale)
    q_act, act_scale, zero_point = quantize_dynamic_uint8_per_channel(activation)
    print(q_act)
    print(act_scale)
    print(zero_point)
    
    print(q_act.shape)
    print(act_scale.shape)
    print(zero_point.shape)