# from conv2d_bfloat16 import conv2d as conv2d_bfloat16
# from conv2d_int8 import conv2d as conv2d_int8
# from linear_bfloat16 import linear as linear_bfloat16
# from linear_int8 import linear as linear_int8

# import im2col
# import quantization





from .conv2d_int8 import conv2d_int8, conv2d_int8_est
from .linear_int8 import linear_int8, linear_int8_est


__all__ = ['conv2d_int8', 'conv2d_int8_est', 'linear_int8', 'linear_int8_est']