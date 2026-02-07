import torch
import approxtorch as at
from torch.autograd import Function




class _bgemm_uint8_ste(Function):
    @staticmethod
    def forward(ctx, qfeature, qweight, lut):
        # feature (N, CKK, L)
        # weight (O, CKK)
        
        
        ctx.save_for_backward(qfeature, qweight)
        q_output = at.backend.ops.bgemm_uint8(qfeature, qweight, lut)
        # output is (N, O, L)
        
        return q_output.to(torch.float)
    
    @staticmethod
    def backward(ctx, grad_output):
        qfeature, qweight = ctx.saved_tensors
        qfeature = qfeature.to(torch.float)
        qweight = qweight.to(torch.float)
        
        grad_qfeature = None
        grad_qweight = None
        if ctx.needs_input_grad[0]:
            grad_qfeature = torch.matmul(qweight.t(), grad_output)
        
        if ctx.needs_input_grad[1]:
            grad_qweight = torch.bmm(grad_output, qfeature.transpose(1, 2)).sum(dim=0)
            
            
        return grad_qfeature, grad_qweight, None
    
    
def bgemm_uint8_ste(qfeature, qweight, lut):
    return _bgemm_uint8_ste.apply(qfeature, qweight, lut)