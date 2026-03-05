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


    
def load_half_custom_grad_lut(file_path):
    # dL/dx use STE
    # only need dL/dw grad lut
    grad_lut_dy = np.loadtxt(file_path, dtype=np.float32)
    grad_lut_dy = torch.tensor(grad_lut_dy)
    grad_lut_dy = grad_lut_dy.view(-1)
    
    return grad_lut_dy

def load_custom_grad_lut(dx_file_path, dw_file_path):
    dx_lut = np.loadtxt(dx_file_path, dtype=np.float32)
    dw_lut = np.loadtxt(dw_file_path, dtype=np.float32)
    dx_lut = torch.tensor(dx_lut)
    dw_lut = torch.tensor(dw_lut)
    dx_lut = dx_lut.view(-1)
    dw_lut = dw_lut.view(-1)
    return dx_lut, dw_lut