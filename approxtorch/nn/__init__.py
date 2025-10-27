
from .Conv2d_int8 import Conv2d_int8_STE, Conv2d_int8_EST, Conv2d_int8_custom
from .conv2d_int8 import conv2d_int8_EST, conv2d_int8_STE, conv2d_int8_custom
from .Depthwise_conv2d_int8 import Depthwise_conv2d_int8_EST, Depthwise_conv2d_int8_STE


from .Linear_int8 import Linear_int8_STE
from .Conv2d_uint8 import Conv2d_uint8_STE
from .Linear_uint8 import Linear_uint8_STE


# int4 series
from .conv2d_int4 import conv2d_int4_exact, conv2d_int4_STE
from .Conv2d_int4 import Conv2d_int4_exact, Conv2d_int4_STE