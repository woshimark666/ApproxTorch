# from conv2d_bfloat16 import conv2d as conv2d_bfloat16
# from conv2d_int8 import conv2d as conv2d_int8
# from linear_bfloat16 import linear as linear_bfloat16
# from linear_int8 import linear as linear_int8

# import im2col
# import quantization


# depthwise conv2d and conv2d 
from .Conv2d_int8 import Conv2d_int8_STE, Conv2d_int8_EST
from .conv2d_int8 import conv2d_int8_EST, conv2d_int8_STE
from .Depthwise_conv2d_int8 import Depthwise_conv2d_int8_EST, Depthwise_conv2d_int8_STE


# not very important, under development
from .Linear_int8 import Linear_int8_STE
from .Conv2d_uint8 import Conv2d_uint8_STE
from .Linear_uint8 import Linear_uint8_STE

from .conv2d_int4 import conv2d_int4_STE
from .linear_int4 import linear_int4_STE

from .Linear_int4 import Linear_int4_STE
from .Conv2d_int4 import Conv2d_int4_STE