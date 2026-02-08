import torch
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import quantizer as Q
from . import bgemm
from . import im2col
import math

class Conv2d_uint8(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int], 
                 lut: torch.Tensor,
                 x_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
                 w_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
                 update_qparams: bool = False,
                 eps: float = 0.05,
                 grad: str = 'ste',
                 grad_dx: torch.Tensor | None = None,
                 grad_dy: torch.Tensor | None = None,
                 bias: torch.Tensor | None = None,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.x_quantizer = x_quantizer
        self.w_quantizer = w_quantizer
        self.grad = grad
        self.update_qparams = update_qparams
        self.eps = eps
        self.qmin = 0
        self.qmax = 255
        
        # lut 
        self.register_buffer('lut', lut)
        # weight
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        # bias
        if isinstance(bias, torch.Tensor):
            self.bias = nn.Parameter(bias)
        elif bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")
        
        if self.x_quantizer[0] == 'static':
            self.register_buffer('scale_x', torch.tensor(()))
            self.register_buffer('zero_x', torch.tensor(()))
        
        if self.w_quantizer[0] == 'static':
            if self.w_quantizer[2] == 'channel':
                self.register_buffer('scale_w', torch.tensor((out_channels,)))
                self.register_buffer('zero_w', torch.tensor((out_channels,)))
            else:
                self.register_buffer('scale_w', torch.tensor(()))
                self.register_buffer('zero_w', torch.tensor(()))
                
        if self.grad != 'ste':
            self.register_buffer('grad_dx', grad_dx)
            self.register_buffer('grad_dy', grad_dy)
            
    
    def __repr__(self):
        return f"Conv2d_uint8(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
            f"kernel_size={self.kernel_size}, grad={self.grad}" \
            f"update_qparams={self.update_qparams}, eps={self.eps})"
      
    def enbale_update_qparams(self):
        self.update_qparams = True
        
    def disable_update_qparams(self):
        self.update_qparams = False
    
    def _update_qparams(self, x: torch.Tensor):
        max_x = torch.max(x)
        min_x = torch.min(x)
        new_scale_x = (max_x - min_x) / 255.
        new_zero_x = - torch.round(min_x / new_scale_x)
        
        if self.w_quantizer[2] == 'tensor':
            max_w = torch.max(self.weight)
            min_w = torch.min(self.weight)
            new_scale_w = (max_w - min_w) / 255.
            new_zero_w = - torch.round(min_w / new_scale_w)
        elif self.w_quantizer[2] == 'channel':
            max_w = torch.amax(self.weight, dim=(1,2,3), keepdim=False)
            min_w = torch.amin(self.weight, dim=(1,2,3), keepdim=False)
            new_scale_w = (max_w - min_w) / 255.
            new_zero_w = - torch.round(min_w / new_scale_w)
        else:
            raise ValueError("Invalid weight quantization method")
        
        new_scale_x = (1-self.eps) * self.scale_x + self.eps * new_scale_x
        new_zero_x = (1-self.eps) * self.zero_x + self.eps * new_zero_x
        new_scale_w = (1-self.eps) * self.scale_w + self.eps * new_scale_w
        new_zero_w = (1-self.eps) * self.zero_w + self.eps * new_zero_w
        
        with torch.no_grad():
            self.scale_x.copy_(new_scale_x)
            self.zero_x.copy_(new_zero_x)
            self.scale_w.copy_(new_scale_w)
            self.zero_w.copy_(new_zero_w)
    
    def forward(self, x: torch.Tensor):
        
        # -1 算shape
        B, C, H, W = x.shape
        kH, kW = _pair(self.kernel_size)
        sH, sW = _pair(self.stride)
        pH, pW = _pair(self.padding)
        dH, dW = _pair(self.dilation)
        O = self.out_channels
        OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        # 0. need update qparams ?
        if self.update_qparams and self.x_quantizer[0] == 'static' and self.w_quantizer[0] == 'static':
            self._update_qparams(x)
        # 1. quantization
        if self.x_quantizer[0] == 'static':
            q_x = Q.asymmetric_static_quantize_uint8_per_tensor(x, 
                self.scale_x, self.zero_x, self.qmin, self.qmax)
        elif self.x_quantizer[0] == 'dynamic':
            pass
        
        if self.w_quantizer[0] == 'static':
            if self.w_quantizer[2] == 'tensor':
                q_w = Q.asymmetric_static_quantize_uint8_per_tensor(self.weight, 
                    self.scale_w, self.zero_w, self.qmin, self.qmax)
            elif self.w_quantizer[2] == 'channel':
                q_w = Q.asymmetric_static_quantize_uint8_per_channel(self.weight, 
                    self.scale_w, self.zero_w, self.qmin, self.qmax)
            else:
                raise ValueError("Invalid weight quantization method")
        elif self.w_quantizer[0] == 'dynamic':
            pass
        
        # 2. im2col 
        q_x = im2col.im2col_uint8(q_x, self.kernel_size, self.stride, self.padding, self.dilation)
        q_w = q_w.view(self.out_channels, -1)
        
        # 3. bgemm using different gradient method
        if self.grad == 'ste':
            output = bgemm.bgemm_uint8_ste(q_x, q_w, self.lut)
        elif self.grad == 'custom':
            pass
        else:
            raise ValueError("Invalid gradient type")
            
        # 4. de-quantization
        output = output.to(torch.float)
        q_w = q_w.to(torch.float)
        q_x = q_x.to(torch.float)
        K = q_w.shape[1]
        # q_w @ 1 → (O,)  每行求和
        qw_sum = q_w.detach().sum(dim=1)  # (O,)

        # 1 @ q_x → (N, 1, L) 或 (N, CKK, L) 按列求和 → (N, 1, L)
        qx_sum = q_x.detach().sum(dim=1, keepdim=True)  # (N, 1, L)
        if self.w_quantizer[2] == 'tensor':
            # per-tensor: s_w, z_w 都是标量
            output = self.scale_x * self.scale_w * (
                output
                - self.zero_x * qw_sum.view(1, -1, 1)      # (1, O, 1)
                - self.zero_w * qx_sum                       # (N, 1, L)
                + self.zero_w * self.zero_x * K
            )

        elif self.w_quantizer[2] == 'channel':
            # per-channel: s_w (O,), z_w (O,)
            s_w = self.scale_w.view(1, -1, 1)   # (1, O, 1)
            z_w = self.zero_w.view(1, -1, 1)    # (1, O, 1)

            output = self.scale_x * s_w * (
                output
                - self.zero_x * qw_sum.view(1, -1, 1)
                - z_w * qx_sum
                + z_w * self.zero_x * K
            )

        # 5. add bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        
        return output.view(B, O, OH, OW)