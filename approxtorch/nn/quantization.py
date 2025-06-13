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