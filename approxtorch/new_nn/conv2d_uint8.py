import torch
import approxtorch as at
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from . import quantization as Q