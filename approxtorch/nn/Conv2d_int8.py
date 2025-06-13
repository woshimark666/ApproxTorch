import torch
from . import conv2d_int8
from torch.nn.modules.utils import _pair

class Conv2d_int8_STE(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 qmethod: tuple[str, str, str]=('dynamic', 'tensor', 'tensor'),
                 qparams: tuple[torch.Tensor, torch.Tensor] | None = None,
                 bias: bool | torch.Tensor = True,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1):
        
        super().__init__()
        self.register_buffer('lut', lut)
        self.register_buffer('qparams', qparams)
        self.in_chahnnels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.qmethod = qmethod
        self.bias = bias
        self.has_bias = bias is not None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        
        if isinstance(self.bias, torch.Tensor):
            self.bias = torch.nn.Parameter(self.bias)
            self.has_bias = True
        elif bias == True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.has_bias = True
        elif bias == False or bias == None:
            self.bias = None
            self.has_bias = False
        else:
            raise ValueError("Invalid bias type")
    
    def __repr__(self):
        return f"Conv2d_int8_STE(in_channels={self.in_chahnnels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, qmethod={self.qmethod}, " \
               f"bias={self.has_bias}, stride={self.stride}, padding={self.padding}, " \
               f"dilation={self.dilation})"
    
    
    def forward(self, x):
        return conv2d_int8.conv2d_int8_STE(x,
                                          self.weight,
                                          self.lut,
                                          self.qmethod,
                                          self.qparams,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation)
    
    
    