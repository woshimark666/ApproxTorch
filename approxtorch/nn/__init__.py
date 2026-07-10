# 当前唯一维护的卷积实现：decoupled 版 Conv2d_int8（整数域 LUT 卷积 + 末端反量化）。
# 旧实现（Conv2d_uint8 / 旧 Conv2d_int8(_v2) / gradual / Conv2dBN / BQSG_float /
# quantizer / BatchNorm2d / fakequant 等）已移入 ./deprecated/，不再从包里导出。
from .Conv2d_int import Conv2d_int8
from . import bgemm
from . import quantization   # 静态 int8/uint8 量化 + 权重 dynamic/grid 量化（原 fakequant）
