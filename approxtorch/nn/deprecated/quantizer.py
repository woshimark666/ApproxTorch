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

def symmetric_static_quantize_int8_per_tensor(x, s, z, qmin, qmax):
    s = s.to(torch.float)
    u = x / s 
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.int8)

def symmetric_static_quantize_int8_per_channel(x, s, z, qmin, qmax):
    s = s.view(-1,1,1,1)
    u = x / s 
    q = torch.round(u).clamp(qmin, qmax)

    return q.to(torch.int8)

def symmetric_dynamic_quantize_int8_per_tensor(x, qmin, qmax):
    absmax = torch.max(torch.abs(x))
    s = absmax / float((qmax - qmin)/2)
    q = torch.round(x / s).clamp(qmin, qmax)
    z = None
    return q.to(torch.int8), s, z

def symmetric_dynamic_quantize_int8_per_channel(x, qmin, qmax):
    absmax = torch.amax(torch.abs(x), dim=(1,2,3), keepdim=True)
    s = absmax / float((qmax - qmin)/2)
    s = s.detach()
    q = torch.round(x / s).clamp(qmin, qmax)
    z = None    
    return q.to(torch.int8), s.view(-1), z






