import torch
from approxtorch.functional import linear_old


class approx_Linear_int8_old(torch.nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 lut,
                 bias = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
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
        return linear_old.apply(x, 
                                self.weight, 
                                self.lut, 
                                self.bias)