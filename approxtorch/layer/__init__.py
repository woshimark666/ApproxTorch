from .Conv2d_int8 import approx_Conv2d_int8
from .Conv2d_int8_naive import approx_Conv2d_int8_naive
from .Conv2d_int8_old import approx_Conv2d_int8_old

from .Linear_int8 import approx_Linear_int8
from .Linear_int8_naive import approx_Linear_int8_naive
from .Linear_int8_old import approx_Linear_int8_old

__all__ = ['approx_Conv2d_int8', 'approx_Conv2d_int8_naive', 'approx_Conv2d_int8_old',
           'approx_Linear_int8', 'approx_Linear_int8_naive', 'approx_Linear_int8_old']