import torch
from approxtorch.functional import conv2d_int8, conv2d_int8_est

class Conv2d_int8(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size, 
                 lut,
                 bias = True,
                 stride = 1,
                 padding = 0,
                 dilation = 1):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.in_chahnnels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.bias = bias
        self.stride = stride[0] if isinstance(stride, tuple) else stride
        self.padding = padding[0] if isinstance(padding, tuple) else padding
        self.dilation = dilation[0] if isinstance(dilation, tuple) else dilation
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size, self.kernel_size))
        
        if bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        elif bias == False or bias == None:
            self.bias = None
        elif isinstance(bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
    
    
    def forward(self, x):
        return conv2d_int8.apply(x, 
                            self.weight, 
                            self.lut,
                            self.bias, 
                            self.stride, 
                            self.padding, 
                            self.dilation)
    
class Conv2d_int8_est(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size, 
                 lut,
                 gradient_lut,
                 bias = True,
                 stride = 1,
                 padding = 0,
                 dilation = 1):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('gradient_lut', gradient_lut)
        self.in_chahnnels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.bias = bias
        self.stride = stride[0] if isinstance(stride, tuple) else stride
        self.padding = padding[0] if isinstance(padding, tuple) else padding
        self.dilation = dilation[0] if isinstance(dilation, tuple) else dilation
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size, self.kernel_size))
        
        if bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        elif bias == False or bias == None:
            self.bias = None
        elif isinstance(bias, torch.Tensor):
            self.bias = torch.nn.Parameter(bias)
    
    def forward(self, x):
        return conv2d_int8_est.apply(x, 
                            self.weight, 
                            self.lut,
                            self.gradient_lut,
                            self.bias, 
                            self.stride, 
                            self.padding, 
                            self.dilation)