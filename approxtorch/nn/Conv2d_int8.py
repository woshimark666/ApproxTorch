import math
import torch
import torch.distributed as dist
from . import quantization
import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import bgemm_int8

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
        # groups: 1（普通卷积）或 depthwise（groups == in == out，即通道
        # multiplier 为 1，MobileNet 系列的用法）；其余分组暂不支持
        if groups != 1 and not (groups == in_channels and out_channels == in_channels):
            raise NotImplementedError(
                "Conv2d_int8 supports groups=1 or depthwise "
                f"(groups == in_channels == out_channels), got groups={groups}, "
                f"in={in_channels}, out={out_channels}")
        self.groups = groups
        self.x_quantizer = x_quantizer
        self.w_quantizer = w_quantizer
        if w_quantizer != 'symmetric':
            raise ValueError("Only symmetric weight quantization is supported")
        if grad not in ('ste', 'lre', 'custom'):
            raise ValueError(f"grad must be 'ste', 'lre' or 'custom', got {grad}")
        if grad == 'custom' and groups != 1:
            raise NotImplementedError(
                "custom gradient currently supports only groups=1")
        self.grad = grad
        self.qmin = -127
        self.qmax = 127
        self.weight_bits = weight_bits
        if not 3 <= weight_bits <= 8:
            raise ValueError(f"weight_bits must be between 3 and 8, got {weight_bits}")
        if not 0.0 <= scale_momentum <= 1.0:
            raise ValueError(
                f"scale_momentum must be between 0 and 1, got {scale_momentum}")
        self.scale_momentum = scale_momentum
        self.update_scale = update_scale  # whether to update scale during training, used for BatchNorm fusion
        
        # lut 
        self.register_buffer('lut', lut)
        # weight（depthwise 时 [O, 1, kH, kW]，与 nn.Conv2d 一致）
        self.weight = nn.Parameter(torch.empty(self.out_channels,
                                               self.in_channels // self.groups,
                                               self.kernel_size[0], self.kernel_size[1]))
        # quantization parameters
        match x_quantizer:
            case 'symmetric':
                self.register_buffer('scale_x', torch.tensor(1.0))
                self.zero_x = None  # 占个位置 没用
            case 'asymmetric':
                raise NotImplementedError("asymmetric quantization for x is not implemented yet")
            case _:
                raise ValueError("Invalid quantization method for x")

        # EMA 权重 scale：语义与 calib.py 的 scale_w 一致（absmax/qmax 的传统
        # dequant scale，per-channel [O]），因此校准 checkpoint 里的
        # {layer}.scale_w 可以直接 load_state_dict 进来。
        self.register_buffer('scale_w', torch.ones(self.out_channels))


        # bias
        if isinstance(bias, torch.Tensor) or bias == True:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        elif bias == False or bias == None:
            self.bias = None
        else:
            raise ValueError("Invalid bias type")

        self.reset_parameters()

        # 用当前权重的 per-channel absmax 初始化；随后 load_state_dict 可用校准值覆盖。
        self._reset_scale_w_from_weight()

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
                if dx is None or dw is None:
                    raise ValueError("dx and dw are required when grad='lre'")
                self.register_buffer('dx', dx)
                self.register_buffer('dw', dw)
            case 'custom':
                if dx is None or dw is None:
                    raise ValueError("dx and dw are required when grad='custom'")
                if not isinstance(dx, torch.Tensor) or not isinstance(dw, torch.Tensor):
                    raise TypeError("custom dx and dw must be torch.Tensor objects")
                if dx.numel() != 256 * 256 or dw.numel() != 256 * 256:
                    raise ValueError(
                        "custom dx and dw must each have 65536 elements "
                        "for grad_lut[x + 128][w + 128]")
                if dx.dtype != torch.float32 or dw.dtype != torch.float32:
                    raise TypeError("custom dx and dw must have dtype torch.float32")
                self.register_buffer('dx', dx.contiguous().view(-1))
                self.register_buffer('dw', dw.contiguous().view(-1))

    def reset_parameters(self):
        # 与 nn.Conv2d 相同的默认初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) \
                     * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"Conv2d_int8(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "\
                f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
                f"x_quantizer={self.x_quantizer}, w_quantizer={self.w_quantizer}, grad={self.grad}, " \
                f"weight_bits={self.weight_bits})"
    

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

    def _update_scale_w(self):
        # 权重 scale 的 EMA。与 _update_scale 同一 momentum 约定。
        # DDP 下权重本身在各 rank 逐位一致，absmax 自然一致，无需 all_reduce。
        with torch.no_grad():
            qmax_w = 2 ** (self.weight_bits - 1) - 1
            reduce_dims = tuple(range(1, self.weight.dim()))
            absmax = self.weight.detach().abs().amax(dim=reduce_dims)   # [O]
            current_scale = torch.where(absmax > 0, absmax / qmax_w,
                                        self.scale_w)
            new_scale = self.scale_momentum * current_scale + (1 - self.scale_momentum) * self.scale_w
            self.scale_w.copy_(new_scale)

    def _reset_scale_w_from_weight(self):
        """按当前权重重置 EMA scale，供初始化和模型转换时使用。"""
        with torch.no_grad():
            qmax_w = 2 ** (self.weight_bits - 1) - 1
            reduce_dims = tuple(range(1, self.weight.dim()))
            absmax = self.weight.detach().abs().amax(dim=reduce_dims)
            self.scale_w.copy_(torch.where(absmax > 0, absmax / qmax_w,
                                           torch.ones_like(absmax)))



    def forward(self, x: torch.Tensor):

        # 0. compute output shape
        B, _, H, W = x.shape
        O, _, kH, kW = self.weight.shape   # weight [O, C//groups, kH, kW]
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
            self._update_scale_w()

        x = quantization.static_quantize_int8(x, self.scale_x, self.qmin, self.qmax)
        # 权重使用 EMA per-channel scale 做普通对称静态量化。
        qmax_w = 2 ** (self.weight_bits - 1) - 1
        w = quantization.static_quantize_int8(
            self.weight, self.scale_w, -qmax_w, qmax_w, ch_axis=0)
        s_w = self.scale_w

        # 2. + 3. im2col + bgemm
        # conv 级 Function：内部 int8 图像 -> im2col_u8 直接喂 LUT kernel；
        # depthwise（groups==C）走专用 dwconv LUT kernel。
        geom = (kernel_size, self.stride, self.padding, self.dilation, self.groups)
        if self.grad == 'lre':
            y = bgemm_int8.conv2d_int8_lre(
                x, w.view(self.out_channels, -1), self.lut, self.dx, self.dw, geom)
        elif self.grad == 'custom':
            y = bgemm_int8.conv2d_int8_custom(
                x, w.view(self.out_channels, -1), self.lut, self.dx, self.dw, geom)
        else:
            y = bgemm_int8.conv2d_int8_ste(
                x, w.view(self.out_channels, -1), self.lut, geom)

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
