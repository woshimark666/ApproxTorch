import torch
import approxtorch as at
from MobileNetv2_cifar10 import MobileNetV2

lut = at.load_lut('exact.txt')
gradient_lut = at.load_gradient_lut('exact_grad.txt')
model = MobileNetV2()

qtype = 'int8'
qmethod = ('static', 'tensor', 'channel')
gradient = 'ste'

model = at.convert_model(model, lut, qtype, qmethod, gradient, gradient_lut,
                         conv_only=True, ignore_first_conv=True)


print(model)







