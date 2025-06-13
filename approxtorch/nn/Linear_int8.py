import torch
from . import linear_int8

class Linear_int8_STE(torch.nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 lut: torch.Tensor,
                 qmethod: str = 'dynamic',
                 qparams: tuple[torch.Tensor, torch.Tensor] | None = None,
                 bias: bool | torch.Tensor = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('qparams', qparams)
        self.qmethod = qmethod
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.has_bias = bias is not None
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        if isinstance(self.bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
            self.has_bias = True
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.has_bias = True
        elif bias == False or bias == None:
            self.bias = None
            self.has_bias = False
        else:
            raise ValueError("Invalid bias type")
    
    def __repr__(self):
        return f"Linear_int8_STE(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"qmethod={self.qmethod}, bias={self.has_bias})"
    
    def forward(self, x):
        return linear_int8.linear_int8_STE(x,
                    self.weight,
                    self.lut,
                    self.qmethod,
                    self.qparams,
                    self.bias)