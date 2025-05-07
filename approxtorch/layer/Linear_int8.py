import torch
from approxtorch.functional import linear_int8, linear_int8_est



class Linear_int8(torch.nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 lut,
                 bias = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        elif bias == False or bias == None:
            self.bias = None
        elif isinstance(bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
    
    def forward(self, x):
        return linear_int8.apply(x, 
                            self.weight, 
                            self.lut, 
                            self.bias)
    
class Linear_int8_est(torch.nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 lut,
                 gradient_lut,
                 bias = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('gradient_lut', gradient_lut)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        elif bias == False or bias == None:
            self.bias = None
        elif isinstance(bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
    
    def forward(self, x):
        return linear_int8_est.apply(x, 
                            self.weight, 
                            self.lut, 
                            self.gradient_lut,
                            self.bias)
                 