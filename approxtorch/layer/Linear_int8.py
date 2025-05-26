import torch
from approxtorch.functional import linear_int8, linear_int8_est, linear_int8_T, linear_int8_est_T



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
        
        if isinstance(self.bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")
    
    def forward(self, x):
        return linear_int8(x, 
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
        
        if isinstance(bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")
    
    def forward(self, x):
        return linear_int8_est(x, 
                            self.weight, 
                            self.lut, 
                            self.gradient_lut,
                            self.bias)

class Linear_int8_T(torch.nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 lut,
                 T_feature,
                 T_weight,
                 bias = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.T_feature = T_feature
        self.T_weight = T_weight
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.has_bias = bias is not None
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.update_T = True
        
        if isinstance(self.bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")
    
    def __repr__(self):
        return f"Linear_int8_STE_T(in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}, T_feature={self.T_feature:.3f}, T_weight={self.T_weight:.3f}), update_T={self.update_T}"
    
    def update_T_func(self, x, weight):
        absmax_feature = torch.abs(x).max().item()
        absmax_weight = torch.abs(weight).max().item()
        self.T_feature = 0.95 * self.T_feature + 0.05 * absmax_feature
        self.T_weight = 0.95 * self.T_weight + 0.05 * absmax_weight
    
    def forward(self, x):
        if self.update_T:
            self.update_T_func(x, self.weight)
        return linear_int8_T(x, 
                            self.weight, 
                            self.lut, 
                            self.T_feature,
                            self.T_weight,
                            self.bias)
        
class Linear_int8_est_T(torch.nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 lut,
                 gradient_lut,
                 T_feature,
                 T_weight,
                 bias = True):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('gradient_lut', gradient_lut)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.T_feature = T_feature
        self.T_weight = T_weight
        self.update_T = True
        self.has_bias = bias is not None
        
        if isinstance(bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")
    def __repr__(self):
        return f"Linear_int8_EST_T(in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}, T_feature={self.T_feature:.3f}, T_weight={self.T_weight:.3f}), update_T={self.update_T}"
    
    def update_T_func(self, x, weight):
        absmax_feature = torch.abs(x).max().item()
        absmax_weight = torch.abs(weight).max().item()
        self.T_feature = 0.95 * self.T_feature + 0.05 * absmax_feature
        self.T_weight = 0.95 * self.T_weight + 0.05 * absmax_weight
    
    def forward(self, x):
        if self.update_T:
            self.update_T_func(x, self.weight)
            
        return linear_int8_est_T(x, 
                            self.weight, 
                            self.lut,
                            self.gradient_lut,
                            self.T_feature,
                            self.T_weight,
                            self.bias)