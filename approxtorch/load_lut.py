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


    
def load_lre_grad_lut(da_file_path, db_file_path):
    
    da_lut = np.loadtxt(da_file_path, dtype=np.float32)
    db_lut = np.loadtxt(db_file_path, dtype=np.float32)
    
    da_lut = torch.tensor(da_lut)
    db_lut = torch.tensor(db_lut)
    da_lut = da_lut.view(-1)
    db_lut = db_lut.view(-1)
    da_lut.requires_grad_(False)
    db_lut.requires_grad_(False)
    return da_lut, db_lut


    
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