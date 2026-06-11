import math
import torch
import torch.distributed as dist
from . import fakequant
import approxtorch as at
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import bgemm

class Conv2d_int8(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 lut: torch.Tensor,
                 x_quantizer:str = 'symmetric',
                 w_quantizer:str = 'symmetric',
                 grad: str = 'ste',
                 dx: torch.Tensor | None = None,
                 dw: torch.Tensor | None = None,
                 coefficient: torch.Tensor | None = None,
                 bias: torch.Tensor | None = None,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 update_scale: bool = True,
                 scale_momentum: float = 0.05,
                 weight_bits: int = 8
         ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        if groups != 1:
            raise NotImplementedError("Conv2d_int8_decoupled only supports groups=1, "
                                      f"got groups={groups}")
        self.groups = groups
        self.x_quantizer = x_quantizer
        self.w_quantizer = w_quantizer
        self.grad = grad
        self.qmin = -127
        self.qmax = 127
        self.weight_bits = weight_bits
        self.scale_momentum = scale_momentum
        self.update_scale = update_scale  # whether to update scale during training, used for BatchNorm fusion
        
        # lut 
        self.register_buffer('lut', lut)
        # weight
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        # quantization parameters
        match x_quantizer:
            case 'symmetric':
                self.register_buffer('scale_x', torch.tensor(1.0))
                self.zero_x = None  # 占个位置 没用
            case 'asymmetric':
                raise NotImplementedError("asymmetric quantization for x is not implemented yet")
            case _:
                raise ValueError("Invalid quantization method for x")
        

        # bias
        if isinstance(bias, torch.Tensor) or bias == True:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")

        self.reset_parameters()

        # 外部给定 bias 时覆盖默认初始化
        if isinstance(bias, torch.Tensor):
            if bias.shape != (self.out_channels,):
                raise ValueError(f"bias must have shape ({self.out_channels},), got {tuple(bias.shape)}")
            with torch.no_grad():
                self.bias.copy_(bias)

        match grad:
            case 'ste':
                pass
            case 'lre':
                self.register_buffer('dx', dx)
                self.register_buffer('dw', dw)
            case 'bqsg64':
                self.register_buffer('coefficient', coefficient)

    def reset_parameters(self):
        # 与 nn.Conv2d 相同的默认初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"Conv2d_int8_decoupled(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"x_quantizer={self.x_quantizer}, w_quantizer={self.w_quantizer}, grad={self.grad})"
    

    def unfreeze_scale(self):
        self.update_scale = True
    def freeze_scale(self):
        self.update_scale = False

    def _update_scale(self, x):
        with torch.no_grad():
            abs_max = x.abs().max()
            # 多卡 DDP 下让 scale 全局同步：先取所有 rank 的全局最大绝对值，
            # 这样每个 rank 算出的 new_scale 完全一致，scale_x 始终保持同步。
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(abs_max, op=dist.ReduceOp.MAX)
            current_scale = abs_max / ((self.qmax - self.qmin) / 2 )
            new_scale = self.scale_momentum * current_scale + (1 - self.scale_momentum) * self.scale_x
            self.scale_x.copy_(new_scale)



    def forward(self, x: torch.Tensor):

        # 0. compute output shape 
        B, C, H, W = x.shape
        O, C, kH, kW = self.weight.shape
        kernel_size = (kH, kW)
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        # 1. do quantization first 
        #   check if we need to update scale:
        if self.training and self.update_scale:
            self._update_scale(x)

        x = fakequant.symmetric_static_quantize_int8_per_tensor(x, self.scale_x, None, self.qmin, self.qmax)
        w, s_w = fakequant.symmetric_dynamic_quantize_int8_per_channel(self.weight, ch_axis=0, bits=self.weight_bits)

        # 2. + 3. im2col + bgemm
        if self.grad == 'lre':
            # conv 级 Function：内部 int8 图像 -> im2col_u8 直接喂 LUT kernel
            # （fp32 unfold 和 kernel 的 fp32->u8 prepass 都不再发生），
            # backward 对 k!=1 走 cuDNN 等价卷积、直接返回图像空间梯度
            y = bgemm.conv2d_int8_lre(
                x, w.view(self.out_channels, -1), self.lut, self.dx, self.dw,
                (kernel_size, self.stride, self.padding, self.dilation))
        else:
            # im2col shape transform
            # 1x1 卷积（无 padding）时 unfold 只是一次 gather 复制：
            # stride=1 直接 view，stride>1 切片再展平，省掉 unfold 的全量复制
            if self.kernel_size == (1, 1) and self.padding == (0, 0):
                if self.stride != (1, 1):
                    x = x[:, :, ::sH, ::sW]
                x = x.flatten(2)  # (N, C, L)
            else:
                x = torch.nn.functional.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # (N, CKK, L)
            w = w.view(self.out_channels, -1) # (O, CKK)

            match self.grad:
                case 'ste':
                    y = bgemm.bgemm_int8_ste(x, w, self.lut)
                    # y [N, O, L]]
                case 'bqsg64':
                    y = bgemm.bgemm_int8_bqsg64(x, w , self.lut, self.coefficient)
                case _:
                    raise ValueError("Invalid gradient method")

        # 4. reshape, de-quantization and bias
        # 先把标量 scale_x 和 per-channel s_w 乘成一个 [O] 向量（s_x、s_w 都
        # 不带梯度），再对 y 做一次 fused 的 y*s+b / y*s，对 y 只扫一遍
        y = y.view(B, O, OH, OW)
        s = (self.scale_x * s_w).view(1, -1, 1, 1)
        if self.bias is not None:
            y = torch.addcmul(self.bias.view(1, -1, 1, 1), y, s)
        else:
            y = y * s

        return y