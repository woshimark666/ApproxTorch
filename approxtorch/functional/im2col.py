import torch
import torch.nn.functional as F


def conv_window(tensor, kernel_size, stride=1, padding=0, dilation=1):
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
