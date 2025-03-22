# ApproxTorch

ApproxTorch is a Pytorch extension for simulating 8-bit signed approximate multipliers in Convolutional Neural Neworks for both traning and inference. Simulation is realized through accessing a pre-defined look-up-table, so theoretically all the 8-bit signed approximate multipliers can be simulated.

## Features

* Support both training and inference.
* Highly optimized CUDA kernel for approximate GEMM (good speed).
* Easy-to-use API just like native Pytorch.

## Requirements

* Linux system is recommended (I didn't test on Windows)
* CUDA version of Pytorch.  Pytorch >= 2.4 and CUDA >=11.8.
* CUDA compatible GPU.
* Ninja is preferred to accelerate compiling. (pip install ninja)

## Installation

```bash
git clone https://github.com/mark531593296/ApproxTorch.git
cd ApproxTorch
pip install .
```

## Quick Start

### 1. Prepare your LUT

First prepare the LUT for your approximate multiplier: a 256*256 matrix saved as a .txt file like below:

```text
A\B | -128  -127  -126 ....   127
----|-----------------------------
-128| 16384 16256  
-127| 16256 16129    ....  
-126| 16128 16002
  . |      .      .
  . |      .        .
  . |      .          .
  . |      .            .
 127|                        16129
```

256 rows and 256 columns, each position represents the multiplicatoin result of the crossponding index A and B. Note that the index is not needed. An example of a exact 8-bit signed multiplier can be found in ./example/exact.txt. (Find your own way to generate this LUT, I think it is pretty easy.)

### 2. Load LUT and define model

```python
import torch
import approxtorch as ap
from approxtorch.layer import approx_Conv2d_int8, approx_Linear_int8
import torch.nn as nn
# we can define a simple LeNet5 with approximate layers.
class approx_lenet5(nn.Module):
    def __init__(self, lut) -> None:
        super(approx_lenet5, self).__init__()
  
        self.conv1 = approx_Conv2d_int8(in_channels=1, out_channels=6, kernel_size=5, lut=lut)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = approx_Conv2d_int8(in_channels=6, out_channels=16, kernel_size=5, lut=lut)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = approx_Linear_int8(in_features=16*4*4, out_features=120, lut=lut)
        self.fc2 = approx_Linear_int8(in_features=120, out_features=84, lut=lut)
        self.fc3 = approx_Linear_int8(in_features=84, out_features=10, lut=lut)
              
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x

device = torch.device('cuda:0') 
# load the LUT using ap.load()
lut = ap.load('exact.txt')
# move lut to gpu
lut = lut.to(device)
# initialize the model
lenet5 = approx_lenet5(lut).to(device)

# inference or training do whatever you like
```

## Documentation

### Load LUT

```Python
import torch
import approxtorch as ap

lut = ap.load_lut(file_path = 'exact.txt')
```

### 2. Convert model

```python
# you can convert a normal model into a approximate model
normal_model = LeNet5()
approximate_model = ap.convert_model(model = normal_model, lut = lut, 
             conv2d_module = ap.layer.approx_Conv2d_int8, linear_module = ap.layer.approx_Linear_int8)
```

### 3. Define approximate layers

```python
# approximate Conv2d
ap_conv2d_layer = ap.layer.approx_Conv2d_int8(in_channels, out_channels, kernel_size, lut, 
                            bias = True, stride =1, padding = 0, dilation = 0)
# approximate Linear
ap_linear_layer = ap.layer.approx_Linear_int8(in_features, out_features, lut, bias = True)
```

## Citation

If you would like to use this in your work, please cite my paper:

```bibtex
@INPROCEEDINGS{10031519,
  author={Ma, Ke and Kimura, Shinji},
  booktitle={2022 19th International SoC Design Conference (ISOCC)}, 
  title={ApproxTorch: An Approximate Multiplier Evaluation Environment for CNNs based on Pytorch}, 
  year={2022},
  volume={},
  number={},
  pages={77-78},
  keywords={Convolution;Graphics processing units;Encoding;Libraries;Behavioral sciences;Approximate multiplier based CNN;Pytorch;Simulation of Approximate Multiplier;GPU Acceleration},
  doi={10.1109/ISOCC56007.2022.10031519}}
```

## Famous quote

[![Readme Quotes](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark&quote=喝可乐不加冰等于没喝)](https://github.com/piyushsuthar/github-readme-quotes)
