# import torch
# import approxtorch as at
# import torch.nn as nn
# from torch.nn.modules.utils import _pair
# from . import quantizer as Q
# import math
# from torch.autograd import Function

# class _conv2d_int8_base(Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, weight: torch.Tensor, 
#                 lut: torch.Tensor, 
#                 grad: str, grad_dx: torch.Tensor | None, grad_dy: torch.Tensor | None,
#                 x_quantizer: tuple[str, str, str], w_quantizer: tuple[str, str, str], 
#                 scale_x: torch.Tensor, zero_x: torch.Tensor | None, 
#                 scale_w: torch.Tensor, zero_w: torch.Tensor | None,
#                 bias, stride, padding, dilation, groups, qmin, qmax
#             ):
        
#         B, C, H, W = x.shape
#         O, C, kH, kW = weight.shape
#         kernel_size = (kH, kW)
#         sH, sW = _pair(stride)
#         pH, pW = _pair(padding)
#         dH, dW = _pair(dilation)
#         OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
#         OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        
#         ctx.x_quantizer = x_quantizer
#         ctx.w_quantizer = w_quantizer
#         ctx.has_bias = bias is not None
#         ctx.kernel_size = kernel_size
#         ctx.output_shape = (B, O, OH, OW)
#         ctx.input_shape = x.shape
#         ctx.stride = stride
#         ctx.padding = padding
#         ctx.dilation = dilation
#         ctx.groups = groups
#         ctx.qmin = qmin
#         ctx.qmax = qmax
        
#         # 1. quantization
#         match x_quantizer:
#             case ('static', 'symmetric', 'tensor'):
#                 q_x = Q.asymmetric_static_quantize_int8_per_tensor(x, 
#                     scale_x, zero_x, qmin, qmax)
#             case ('static', 'asymmetric', 'tensor'):
#                 raise ValueError("Asymmetric quantization is not supported yet")
#             case _:
#                 raise ValueError("Invalid input quantization method")
        
#         # 2. im2col
#         q_x = at.backend.ops.im2col_int8(q_x, kernel_size, stride, padding, dilation)
#         q_w = q_w.view(O, -1)
        
#         match grad:
#             case 'ste':
#                 ctx.save_for_backward(x, weight)
#             case 'int_ste':
#                 ctx.save_for_backward(q_x, q_w, scale_x, zero_x, scale_w, zero_w)
#             case _:
#                 raise ValueError("Invalid gradient type")
        
#         # 3. bgemm
#         output = at.backend.ops.bgemm_int8(q_x, q_w, lut)
#         output = output.to(torch.float)
#         q_x = q_x.to(torch.float)
#         q_w = q_w.to(torch.float)
#         # 4. de-quantization
#         match w_quantizer[2]:
#             case 'tensor':
#                 # q_x (N, CKK, L)
#                 # q_w (O, CKK)
#                 # output (N, O, L)
#                 pass
#             case 'channel':
#                 # q_x (N, CKK, L)
#                 # q_w (O, CKK)
#                 # output (N, O, L)
#                 # scale_w (O,) zero_w (O,)
#                 output = output - zero_x * q_w.sum(dim=1).view(1, -1, 1) - \
#                     zero_w.view(1, O, 1) * q_x.sum(dim=1, keepdim=True) + \
#                     zero_x * zero_w.view(1, O, 1) * C * kH * kW
#                 output = output * scale_x * scale_w.view(-1, O, 1)
#             case _:
#                 raise ValueError("Invalid weight quantization method")
#         output = output.view(B, O, OH, OW)
#         # 5. add bias
#         if bias is not None:
#             output = output + bias.view(1, -1, 1)
        
#         return output



# class _conv2d_int8_ste(_conv2d_uint8_base):
#     @staticmethod
#     def backward(ctx, upstream_grad):
#         grad_x, grad_weight, grad_bias = None, None, None
#         x, weight = ctx.saved_tensors
#         if ctx.has_bias and ctx.needs_input_grad[10]:
#             grad_bias = upstream_grad.sum(dim=(0, 2, 3))
            
            
#         grad_x = torch.nn.grad.conv2d_input(x.shape, weight, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
#         grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, upstream_grad, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
#         return grad_x, grad_weight, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None


# class _conv2d_int8_int_ste(_conv2d_int8_base):
#     @staticmethod
#     def backward(ctx, upstream_grad):
#         grad_x, grad_weight, grad_bias,  = None, None, None
#         q_x, q_w, scale_x, zero_x, scale_w, zero_w = ctx.saved_tensors
#         q_x = q_x.to(torch.float)
#         q_w = q_w.to(torch.float)
        
#         B, O, OH, OW = ctx.output_shape
#         B, C, H, W = ctx.input_shape
#         kH, kW = ctx.kernel_size
#         L = OH * OW
#         upstream_grad = upstream_grad.view(B,O,L)
        
#         if ctx.has_bias and ctx.needs_input_grad[10]:
#             grad_bias = upstream_grad.sum(dim=(0, 2))
            
        
#         if ctx.w_quantizer[2] == 'tensor':
#             # compute grad_x
#             # term1 = torch.einsum('nol,ok->nkl', upstream_grad, q_w)
#             dx_term1 = torch.matmul(q_w.t(), upstream_grad)  
#             dx_term2 = - zero_w * upstream_grad.sum(dim=1, keepdim=True)
#             grad_x = (dx_term1 + dx_term2) * scale_w # (N, CKK, L)
#             grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation)
#             # grad_x shape is (B, C, H, W)
            
#             # compute grad_weight
#             # dy_term1 = torch.einsum('nol,nkl->ok', upstream_grad, q_x)
#             dy_term1 = torch.bmm(upstream_grad, q_x.transpose(1, 2)).sum(dim=0) 
#             dy_term2 = - zero_x * upstream_grad
#             dy_term2 = dy_term2.sum(dim=(0, 2)).view(-1, 1)
#             grad_weight = (dy_term1 + dy_term2) * scale_x # (O, CKK)
#             grad_weight = grad_weight.view(O, C, kH, kW)
#             # grad_weight shape is (O, C, kH, kW)
            
#         if ctx.w_quantizer[2] == 'channel':
#             # compute grad_x
#             # term1 = torch.einsum('nol,ok->nkl', upstream_grad, q_w)
#             upstream_grad_scaled = upstream_grad.view(B, O, L) * scale_w.view(1, O, 1)
#             dx_term1 = torch.matmul(q_w.t(), upstream_grad_scaled)  # (N, K, L)
#             dx_term2 = - (zero_w.view(1, O, 1) * upstream_grad_scaled).sum(dim=1, keepdim=True) # (N, 1, L)
#             grad_x = dx_term1 + dx_term2
#             grad_x = torch.nn.functional.fold(grad_x, (H, W), ctx.kernel_size, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation)
#             # grad_x shape is (B, C, H, W)
            
#             # compute grad_weight
#             # dy_term1 = torch.einsum('nol,nkl->ok', upstream_grad, q_x)
#             dy_term1 = torch.bmm(upstream_grad, q_x.transpose(1, 2)).sum(dim=0) 
#             dy_term2 = - zero_x * upstream_grad
#             dy_term2 = dy_term2.sum(dim=(0, 2)).view(-1, 1)
#             grad_weight = (dy_term1 + dy_term2) * scale_x # (O, CKK)
#             grad_weight = grad_weight.view(O, C, kH, kW)
#             # grad_weight shape is (O, C, kH, kW)
        
#         return grad_x, grad_weight, None, None, None, None, None, None, None, None, grad_bias, None, None, None, None, None, None

# def conv2d_uint8(x, weight, lut, grad, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax):
    
#     match grad:
#         case 'ste':
#             return _conv2d_uint8_ste.apply(x, weight, lut, grad, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax)
#         case 'int_ste':
#             return _conv2d_uint8_int_ste.apply(x, weight, lut, grad, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w, bias, stride, padding, dilation, groups, qmin, qmax)
#         case _:
#             raise ValueError("Invalid gradient type")


# class Conv2d_uint8(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int | tuple[int, int], 
#                  lut: torch.Tensor,
#                  x_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
#                  w_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
#                  update_qparams: bool = False,
#                  eps: float = 0.05,
#                  grad: str = 'ste',
#                  grad_dx: torch.Tensor | None = None,
#                  grad_dy: torch.Tensor | None = None,
#                  bias: torch.Tensor | None = None,
#                  stride: int | tuple[int, int] = 1,
#                  padding: int | tuple[int, int] = 0,
#                  dilation: int | tuple[int, int] = 1,
#                  groups: int = 1):
        
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.x_quantizer = x_quantizer
#         self.w_quantizer = w_quantizer
#         self.grad = grad
#         self.update_qparams = update_qparams
#         self.eps = eps
#         self.qmin = 0
#         self.qmax = 255
        
#         # lut 
#         self.register_buffer('lut', lut)
#         # weight
#         self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
#         # bias
#         if isinstance(bias, torch.Tensor):
#             self.bias = nn.Parameter(bias)
#         elif bias == True:
#             self.bias = nn.Parameter(torch.Tensor(self.out_channels))
#         elif bias == False or bias == None:
#             self.bias = None
#         else:
#             raise ValueError("Invalid bias type")
        
#         if self.x_quantizer[0] == 'static':
#             self.register_buffer('scale_x', torch.tensor(1., dtype=torch.float32))
#             self.register_buffer('zero_x', torch.tensor(0., dtype=torch.float32))
        
#         if self.w_quantizer[0] == 'static':
#             if self.w_quantizer[2] == 'channel':
#                 self.register_buffer('scale_w', torch.ones(self.out_channels, ))
#                 self.register_buffer('zero_w', torch.zeros(self.out_channels, ))
#             else:
#                 self.register_buffer('scale_w', torch.tensor(1., dtype=torch.float32))
#                 self.register_buffer('zero_w', torch.tensor(0., dtype=torch.float32))
                
#         if self.grad != 'ste':
#             self.register_buffer('grad_dx', grad_dx)
#             self.register_buffer('grad_dy', grad_dy)
            
    
#     def __repr__(self):
#         return f"Conv2d_uint8(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
#             f"kernel_size={self.kernel_size}, grad={self.grad}, " \
#             f"update_qparams={self.update_qparams}, eps={self.eps})"
      
#     def enbale_update_qparams(self):
#         self.update_qparams = True
        
#     def disable_update_qparams(self):
#         self.update_qparams = False
    
#     def _update_qparams(self, x: torch.Tensor):
#         max_x = torch.max(x)
#         min_x = torch.min(x)
#         new_scale_x = (max_x - min_x) / 255.
#         new_zero_x = - torch.round(min_x / new_scale_x)
        
#         if self.w_quantizer[2] == 'tensor':
#             max_w = torch.max(self.weight)
#             min_w = torch.min(self.weight)
#             new_scale_w = (max_w - min_w) / 255.
#             new_zero_w = - torch.round(min_w / new_scale_w)
#         elif self.w_quantizer[2] == 'channel':
#             max_w = torch.amax(self.weight, dim=(1,2,3), keepdim=False)
#             min_w = torch.amin(self.weight, dim=(1,2,3), keepdim=False)
#             new_scale_w = (max_w - min_w) / 255.
#             new_zero_w = - torch.round(min_w / new_scale_w)
#         else:
#             raise ValueError("Invalid weight quantization method")
        
#         new_scale_x = (1-self.eps) * self.scale_x + self.eps * new_scale_x
#         new_zero_x = (1-self.eps) * self.zero_x + self.eps * new_zero_x
#         new_scale_w = (1-self.eps) * self.scale_w + self.eps * new_scale_w
#         new_zero_w = (1-self.eps) * self.zero_w + self.eps * new_zero_w
        
#         with torch.no_grad():
#             self.scale_x.copy_(new_scale_x)
#             self.zero_x.copy_(new_zero_x)
#             self.scale_w.copy_(new_scale_w)
#             self.zero_w.copy_(new_zero_w)
    
#     def forward(self, x: torch.Tensor):
        
#         if self.update_qparams and self.x_quantizer[0] == 'static' and self.w_quantizer[0] == 'static':
#             self._update_qparams(x)
        
        
#         output = conv2d_uint8(x, self.weight, self.lut, self.grad, self.x_quantizer, self.w_quantizer,
#                                 self.scale_x, self.zero_x, self.scale_w, self.zero_w, self.bias,
#                                 self.stride, self.padding, self.dilation, self.groups, self.qmin, self.qmax)
        
#         return output
    
    

# # class Conv2d_uint8(nn.Module):
# #     def __init__(self,
# #                  in_channels: int,
# #                  out_channels: int,
# #                  kernel_size: int | tuple[int, int], 
# #                  lut: torch.Tensor,
# #                  x_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
# #                  w_quantizer: tuple[str, str, str] = ('static', 'asymmetric', 'tensor'),
# #                  update_qparams: bool = False,
# #                  eps: float = 0.05,
# #                  grad: str = 'ste',
# #                  grad_dx: torch.Tensor | None = None,
# #                  grad_dy: torch.Tensor | None = None,
# #                  bias: torch.Tensor | None = None,
# #                  stride: int | tuple[int, int] = 1,
# #                  padding: int | tuple[int, int] = 0,
# #                  dilation: int | tuple[int, int] = 1,
# #                  groups: int = 1):
        
# #         super().__init__()
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.kernel_size = _pair(kernel_size)
# #         self.stride = _pair(stride)
# #         self.padding = _pair(padding)
# #         self.dilation = _pair(dilation)
# #         self.groups = groups
# #         self.x_quantizer = x_quantizer
# #         self.w_quantizer = w_quantizer
# #         self.grad = grad
# #         self.update_qparams = update_qparams
# #         self.eps = eps
# #         self.qmin = 0
# #         self.qmax = 255
        
# #         # lut 
# #         self.register_buffer('lut', lut)
# #         # weight
# #         self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
# #         # bias
# #         if isinstance(bias, torch.Tensor):
# #             self.bias = nn.Parameter(bias)
# #         elif bias == True:
# #             self.bias = nn.Parameter(torch.Tensor(self.out_channels))
# #         elif bias == False or bias == None:
# #             self.bias = None
# #         else:
# #             raise ValueError("Invalid bias type")
        
# #         if self.x_quantizer[0] == 'static':
# #             self.register_buffer('scale_x', torch.tensor(1., dtype=torch.float32))
# #             self.register_buffer('zero_x', torch.tensor(0., dtype=torch.float32))
        
# #         if self.w_quantizer[0] == 'static':
# #             if self.w_quantizer[2] == 'channel':
# #                 self.register_buffer('scale_w', torch.tensor((self.out_channels,)))
# #                 self.register_buffer('zero_w', torch.tensor((self.out_channels,)))
# #             else:
# #                 self.register_buffer('scale_w', torch.tensor(1., dtype=torch.float32))
# #                 self.register_buffer('zero_w', torch.tensor(0., dtype=torch.float32))
                
# #         if self.grad != 'ste':
# #             self.register_buffer('grad_dx', grad_dx)
# #             self.register_buffer('grad_dy', grad_dy)
            
    
# #     def __repr__(self):
# #         return f"Conv2d_uint8(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
# #             f"kernel_size={self.kernel_size}, grad={self.grad}, " \
# #             f"update_qparams={self.update_qparams}, eps={self.eps})"
      
# #     def enbale_update_qparams(self):
# #         self.update_qparams = True
        
# #     def disable_update_qparams(self):
# #         self.update_qparams = False
    
# #     def _update_qparams(self, x: torch.Tensor):
# #         max_x = torch.max(x)
# #         min_x = torch.min(x)
# #         new_scale_x = (max_x - min_x) / 255.
# #         new_zero_x = - torch.round(min_x / new_scale_x)
        
# #         if self.w_quantizer[2] == 'tensor':
# #             max_w = torch.max(self.weight)
# #             min_w = torch.min(self.weight)
# #             new_scale_w = (max_w - min_w) / 255.
# #             new_zero_w = - torch.round(min_w / new_scale_w)
# #         elif self.w_quantizer[2] == 'channel':
# #             max_w = torch.amax(self.weight, dim=(1,2,3), keepdim=False)
# #             min_w = torch.amin(self.weight, dim=(1,2,3), keepdim=False)
# #             new_scale_w = (max_w - min_w) / 255.
# #             new_zero_w = - torch.round(min_w / new_scale_w)
# #         else:
# #             raise ValueError("Invalid weight quantization method")
        
# #         new_scale_x = (1-self.eps) * self.scale_x + self.eps * new_scale_x
# #         new_zero_x = (1-self.eps) * self.zero_x + self.eps * new_zero_x
# #         new_scale_w = (1-self.eps) * self.scale_w + self.eps * new_scale_w
# #         new_zero_w = (1-self.eps) * self.zero_w + self.eps * new_zero_w
        
# #         with torch.no_grad():
# #             self.scale_x.copy_(new_scale_x)
# #             self.zero_x.copy_(new_zero_x)
# #             self.scale_w.copy_(new_scale_w)
# #             self.zero_w.copy_(new_zero_w)
    
# #     def forward(self, x: torch.Tensor):
        
# #         # -1 算shape
# #         B, C, H, W = x.shape
# #         kH, kW = _pair(self.kernel_size)
# #         sH, sW = _pair(self.stride)
# #         pH, pW = _pair(self.padding)
# #         dH, dW = _pair(self.dilation)
# #         O = self.out_channels
# #         OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
# #         OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
# #         # 0. need update qparams ?
# #         if self.update_qparams and self.x_quantizer[0] == 'static' and self.w_quantizer[0] == 'static':
# #             self._update_qparams(x)
        
        
# #         # 1. quantization
# #         if self.x_quantizer[0] == 'static':
# #             q_x = Q.asymmetric_static_quantize_uint8_per_tensor(x, 
# #                 self.scale_x, self.zero_x, self.qmin, self.qmax)
# #         elif self.x_quantizer[0] == 'dynamic':
# #             pass
        
# #         if self.w_quantizer[0] == 'static':
# #             if self.w_quantizer[2] == 'tensor':
# #                 q_w = Q.asymmetric_static_quantize_uint8_per_tensor(self.weight, 
# #                     self.scale_w, self.zero_w, self.qmin, self.qmax)
# #             elif self.w_quantizer[2] == 'channel':
# #                 q_w = Q.asymmetric_static_quantize_uint8_per_channel(self.weight, 
# #                     self.scale_w, self.zero_w, self.qmin, self.qmax)
# #             else:
# #                 raise ValueError("Invalid weight quantization method")
# #         elif self.w_quantizer[0] == 'dynamic':
# #             pass
        
# #         # 2. im2col 
# #         q_x = im2col.im2col_uint8(q_x, self.kernel_size, self.stride, self.padding, self.dilation)
# #         q_w = q_w.view(self.out_channels, -1)
        
# #         # 3. bgemm using different gradient method
# #         if self.grad == 'ste':
# #             output = bgemm.bgemm_uint8_ste(q_x, q_w, self.lut)
# #         elif self.grad == 'custom':
# #             pass
# #         else:
# #             raise ValueError("Invalid gradient type")
            

# #         # 4. de-quantization
# #         q_x = q_x.to(torch.float)
# #         q_w = q_w.to(torch.float)
# #         match self.w_quantizer[2]:
# #             case 'tensor':
# #                 output = output - self.zero_x * q_w.sum(dim=1).view(1, -1, 1) - \
# #                     self.zero_w * q_x.sum(dim=1, keepdim=True) + self.zero_x * self.zero_w * C * kH * kW
# #                 output = output * self.scale_x * self.scale_w
# #             case 'channel':
# #                 pass
# #             case _:
# #                 raise ValueError("Invalid weight quantization method")
        
        
# #         # 5. add bias
# #         if self.bias is not None:
# #             output = output + self.bias.view(1, -1, 1)
        
# #         return output.view(B, O, OH, OW)
    
    