# from conv2d_bfloat16 import conv2d as conv2d_bfloat16
# from conv2d_int8 import conv2d as conv2d_int8
# from linear_bfloat16 import linear as linear_bfloat16
# from linear_int8 import linear as linear_int8

# import im2col
# import quantization



from .Conv2d_int8 import Conv2d_int8_STE
from .Linear_int8 import Linear_int8_STE

__all__ = ['Conv2d_int8_STE', 'Linear_int8_STE']