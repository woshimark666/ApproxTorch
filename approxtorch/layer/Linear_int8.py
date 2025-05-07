import torch
from approxtorch.functional import linear



class approx_Linear_int8(torch.nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 lut,
                 gradient_lut,
                 bias = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('gradient_lut', gradient_lut)
        # self.lut = lut
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
    
    def forward(self, x):
        return linear.apply(x, 
                            self.weight, 
                            self.lut, 
                            self.bias)