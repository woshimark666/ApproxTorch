import torch
import approxtorch as ap


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
    
device = torch.device('cuda')
A = torch.randn(128, 128, device=device)
B = torch.randn(128, 128, device=device)

qA, sA, zA = quantize_dynamic_uint8(A, (0,1))
qB, sB, zB = quantize_dynamic_uint8(B, (0))

print(sB, zB)
right_output = torch.matmul((qA - zA), (qB - zB))

# another method 
output = torch.matmul(qA, qB)
output = output - zA * qB.sum(dim=0, keepdim=True) - zB * qA.sum(dim=1, keepdim=True) + \
            qA.shape[1] * zA * zB

print(output - right_output)





