import torch
import torch.nn.functional as F
from typing import Tuple, Union
from torch.nn.modules.utils import _pair

def conv_window(tensor: torch.Tensor, 
                kernel_size: Union[int, Tuple[int, int]], 
                stride: Union[int, Tuple[int, int]] = 1, 
                padding: Union[int, Tuple[int, int]] = 0, 
                dilation: Union[int, Tuple[int, int]] = 1):
    """
    Convert a tensor into a windowed tensor for convolution.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolution kernel
        stride (Union[int, Tuple[int, int]]): Stride for the convolution
        padding (Union[int, Tuple[int, int]]): Padding for the convolution
        dilation (Union[int, Tuple[int, int]]): Dilation for the convolution

    Returns:
        torch.Tensor: Windowed tensor of shape (B*L, CKK)
    """
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    tensor = F.unfold(tensor, kernel_size, stride=stride, padding=padding, dilation=dilation)   #(B, CKK, L)
    tensor = tensor.transpose(1, 2) # (B, L, CKK)
    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[2]) # (B*L, CKK)
    return tensor

def conv_weight(weight): #  input weight is (O, C, K, K)
    weight = weight.view(weight.shape[0], -1) # (O, CKK)
    weight = weight.permute(1, 0) # (CKK, O)
    weight = weight.contiguous()
    return weight  # (CKK, O)
