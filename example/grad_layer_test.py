import torch
import approxtorch as at


lut = at.load_lut('exact.txt')
grad_lut = at.load_gradient_lut('exact_grad.txt')


scale_feature = torch.ones(size=(), dtype=torch.float32)
scale_weight = torch.ones(size=(), dtype=torch.float32)




