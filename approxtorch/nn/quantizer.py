import torch
import torch.nn as nn
from torch.autograd import Function

def asymmetric_static_quantize_uint8_per_tensor(x, s, z, qmin, qmax):
    
    s = s.to(torch.float)
    z = z.to(torch.float)  
    
    u = x / s + z
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.uint8)


def asymmetric_static_quantize_uint8_per_channel(x, s, z, qmin, qmax):
    
    s = s.to(torch.float)
    z = z.to(torch.float)  
    
    s = s.view(-1,1,1,1)
    z = z.view(-1,1,1,1)
    
    u = x / s + z
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.uint8)







# static asymmetric quantization for int8 per tensor
class _symmetric_static_quantize_int8_per_tensor(Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                s: torch.Tensor, 
                z: torch.Tensor,
                qmin: int, 
                qmax: int
                ):


        s = s.to(torch.float)
        z = z.to(torch.float)  
        
        u = x / s + z
        q = torch.round(u).clamp(qmin, qmax)

        ctx.save_for_backward(u, s)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return q.to(torch.uint8)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        u, s = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        gate = (u >= qmin) & (u <= qmax)
        gate = gate.to(dtype=grad_out.dtype)

        grad_x = grad_out * gate / s

        # only return grad for x
        return grad_x, None, None, None, None, None