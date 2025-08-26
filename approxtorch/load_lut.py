import numpy as np
import torch

def load_lut(file_path):
    # Load the txt file as a numpy array
    lut_array = np.loadtxt(file_path, dtype=np.int32).reshape(256, 256)
    
    # Convert the numpy array to a PyTorch tensor
    lut_tensor = torch.tensor(lut_array)
    lut_tensor = lut_tensor.view(65536).to(torch.int)
    lut_tensor.requires_grad_(False)
    return lut_tensor


def load_gradient_lut(file_path):
    lut_array = np.loadtxt(file_path, dtype=np.float32).reshape(256, 2)
    lut_tensor = torch.tensor(lut_array)
    lut_tensor.requires_grad_(False)
    
    grad_lut_dx = lut_tensor[:, 0]
    grad_lut_dy = lut_tensor[:, 1]
    grad_lut_dx = grad_lut_dx.contiguous()
    grad_lut_dy = grad_lut_dy.contiguous()
    grad_lut = (grad_lut_dx, grad_lut_dy)
    return grad_lut   
