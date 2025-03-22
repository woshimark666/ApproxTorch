import torch


# quantization from float to int8

def quantize_tensor(x: torch.Tensor):  # scale is T/127
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127.
    x = torch.round(x / scale)
    x = torch.clamp(x, -127, 127)
    return x, scale

def quantize_activiton_tensor_int8(x: torch.Tensor, T):  # scale is T/127, T is the trancation threshold
    scale = T / 127
    x = torch.round(x / scale)
    x = torch.clamp(x, -127, 127)
    return x, scale
    
def quantize_weight_tensor_int8(x: torch.Tensor):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127
    x = torch.round(x / scale)
    x = torch.clamp(x, -127, 127)
    return x, scale

def dequantize_tensor(x, scale):
    
    if(x.dtype != torch.float):
        x = x.to(torch.float)
        
    return x * scale
    
    