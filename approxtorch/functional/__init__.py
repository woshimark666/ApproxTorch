# from conv2d_bfloat16 import conv2d as conv2d_bfloat16
# from conv2d_int8 import conv2d as conv2d_int8
# from linear_bfloat16 import linear as linear_bfloat16
# from linear_int8 import linear as linear_int8

# import im2col
# import quantization

# __all__ = ['conv2d_bfloat16', 'conv2d_int8', 'linear_bfloat16', 'linear_int8', 'im2col', 'quantization']\



from .conv2d_int8 import conv2d 
from .conv2d_int8 import conv2d_int8
from .conv2d_int8_naive import conv2d_naive
from .conv2d_int8_old import conv2d_old 

from .linear_int8 import linear
from .linear_int8 import linear_int8
from .linear_int8_naive import linear_naive 
from .linear_int8_old import linear_old 


__all__ = ['conv2d', 'conv2d_int8', 'conv2d_naive', 'conv2d_old',
           'linear', 'linear_int8', 'linear_naive', 'linear_old']