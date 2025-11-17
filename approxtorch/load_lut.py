import numpy as np
import torch
from typing import Literal

def load_lut(file_path, qtype: Literal['int8', 'int4', 'uint8'] = 'int8'):
    
    if qtype == 'int8' or qtype == 'uint8':
        shape = 256
    else:
        shape = 16
    
    # Load the txt file as a numpy array
    lut_array = np.loadtxt(file_path, dtype=np.int32).reshape(shape, shape)
    
    # Convert the numpy array to a PyTorch tensor
    lut_tensor = torch.tensor(lut_array)
    lut_tensor = lut_tensor.view(shape * shape).to(torch.int)
    lut_tensor.requires_grad_(False)
    return lut_tensor



def load_double_gradient_lut(file0, file1, qtype: Literal['int8', 'int4'] = 'int8'):
    if qtype == 'int8':
        shape = 256
    else:
        shape = 16
    
    dx = np.loadtxt(file0, dtype=np.float32).reshape(shape, shape)
    dy = np.loadtxt(file1, dtype=np.float32).reshape(shape, shape)
    
    dx = torch.tensor(dx)
    dy = torch.tensor(dy)

    dx.requires_grad_(False)
    dy.requires_grad_(False)
    
    return (dx, dy)

    
    
    
    
def load_lre_grad_lut(file_path):
    lut_array = np.loadtxt(file_path, dtype=np.float32).reshape(256, 2)
    lut_tensor = torch.tensor(lut_array)
    lut_tensor.requires_grad_(False)
    
    grad_lut_dx = lut_tensor[:, 0]
    grad_lut_dy = lut_tensor[:, 1]
    grad_lut_dx = grad_lut_dx.contiguous()
    grad_lut_dy = grad_lut_dy.contiguous()
    grad_lut = (grad_lut_dx, grad_lut_dy)
    return grad_lut   

def load_custom_grad_lut(file_path):
    if isinstance(file_path, tuple) and len(file_path) == 2 and all(isinstance(f, str) for f in file_path):
        # 如果 file_path 是一个由两个字符串组成的 tuple
        grad_a = np.loadtxt(file_path[0], dtype=np.float32)
        grad_b = np.loadtxt(file_path[1], dtype=np.float32)
        grad_a = torch.tensor(grad_a)
        grad_b = torch.tensor(grad_b)
        grad_a = grad_a.view(-1)
        grad_b = grad_b.view(-1)
        return grad_a, grad_b
    else:
        grad_a = np.loadtxt(file_path+'_grad_a.txt', dtype=np.float32)
        grad_b = np.loadtxt(file_path+'_grad_b.txt', dtype=np.float32)
        grad_a = torch.tensor(grad_a)
        grad_b = torch.tensor(grad_b)
        grad_a = grad_a.view(-1)
        grad_b = grad_b.view(-1)
        return grad_a, grad_b