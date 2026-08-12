# ApproxTorch

**ApproxTorch** is a PyTorch extension for simulating **approximate multipliers** inside Convolutional Neural Networks — for both **training (QAT)** and **inference** — entirely on the GPU.

Approximate multipliers are hardware circuits that trade a small amount of arithmetic accuracy for large savings in power and area. Before taping out such hardware, you need to know how much CNN accuracy you lose — and how much you can recover by retraining. ApproxTorch answers this question by replacing every multiplication inside convolution layers with a **look-up table (LUT)** of your multiplier's actual behavior, computed by highly optimized custom CUDA kernels.

Because the simulation is purely LUT-driven, **any 8-bit (signed or unsigned) approximate multiplier can be simulated** — no need to write kernel code for each design.

```text
 FP32 input ──► quantize (int8) ──► im2col ──► LUT-based approximate GEMM ──► dequantize ──► FP32 output
                                                  (custom CUDA kernel)
```

## Features

- 🚀 **GPU-accelerated**: hand-written CUDA kernels for LUT-based approximate batched GEMM — fast enough to retrain networks like ResNet on ImageNet-scale data.
- 🎯 **Train *and* infer**: full autograd support, so you can do approximate-multiplier-aware retraining (QAT), not just evaluation.
- 🧮 **Any 8-bit multiplier**: behavior is defined entirely by a 256×256 LUT text file.
- 🔁 **Gradient estimators** for backpropagating through the non-differentiable LUT:
  - **STE** — straight-through estimator (default)
  - **LRE** — linear-regression-estimated gradient LUTs
  - **Custom** — one derivative for every quantized `(x, w)` pair
- ⚖️ **EMA quantization**: per-tensor activation scale and per-channel weight scale are EMA-updated during training, with arbitrary weight bit-width (**3–8 bit**) in the decoupled layer.
- 🔌 **One-line model conversion**: `at.to_qat_int8(model, lut)` swaps every `nn.Conv2d` for an approximate one.
- 🖥️ **Multi-GPU (DDP) support**: activation scales are synchronized across ranks automatically.

## Requirements

- Linux (Windows is untested)
- Python ≥ 3.10
- PyTorch ≥ 2.4 built with CUDA ≥ 11.8
- A CUDA-capable GPU (kernels are compiled with `-arch=native` for your local GPU)
- `ninja` is recommended for faster compilation (`pip install ninja`)

## Installation

```bash
git clone https://github.com/mark531593296/ApproxTorch.git
cd ApproxTorch
pip install .
```

Or, for an editable install with [uv](https://github.com/astral-sh/uv):

```bash
./install.sh        # runs: uv pip install -e . --no-build-isolation
```

The CUDA extension is compiled during installation, which takes a few minutes.

## Quick Start

### 1. Prepare the multiplier LUT

Describe your approximate multiplier as a plain-text file containing a **256×256 matrix** of integers: entry *(i, j)* is the output of your multiplier for inputs `A = i − 128` and `B = j − 128` (signed int8, row-major, indices not included in the file):

```text
16384  16256  16128  ...        # A = -128:  (-128)×(-128), (-128)×(-127), ...
16256  16129  ...               # A = -127
  ...
                     ...  16129 # A =  127
```

An exact-multiplier example is provided at `test/exact_int8.txt`; LUTs of several published approximate multipliers are in `test/` as well (e.g. `mul8u_syn1.txt`, `venka.txt`, `zhang.txt`).

### 2. Convert your model and train

```python
import torch
import approxtorch as at
from torchvision.models import resnet18

device = torch.device('cuda')

# 1. load the LUT and convert it to the float32 format required by CUDA kernels
lut = at.load_lut.load_lut('test/exact_int8.txt', qtype='int8').to(
    device=device, dtype=torch.float32)

# 2. take any FP32 model and replace its Conv2d layers with approximate ones
model = resnet18(num_classes=10)
model = at.to_qat_int8(
    model, lut,
    grad='ste',               # gradient estimator: 'ste' | 'lre' | 'custom'
    conv_only=True,           # only convert Conv2d layers
    ignore_first_conv=True,   # keep the first conv exact (common QAT practice)
    weight_bits=8,            # weight quantization bit-width, 3–8
).to(device)

# 3. business as usual — training and inference work like any PyTorch model
out = model(torch.randn(8, 3, 32, 32, device=device))
out.sum().backward()
```

During training, both activation and per-channel weight scales are updated as exponential moving averages (controlled by `scale_momentum`). Call `model.eval()` (or `freeze_scale()` on a layer) to stop scale updates for inference.

## Gradient Estimators

An approximate multiplier is a discrete function, so its true gradient is not defined. ApproxTorch provides three ways to backpropagate through it:

| `grad=`   | Idea | Extra inputs needed |
|-----------|------|---------------------|
| `'ste'`   | Pretend the multiply was exact (straight-through estimator) | none |
| `'lre'`   | Fit a line to each row/column of the LUT; use slopes as gradients | `dx`, `dw` gradient LUTs |
| `'custom'` | Use one derivative for each quantized `(x, w)` pair | 256×256 `dx`, `dw` gradient LUTs |

The helper module `approxtorch.grad_lut` generates these extra inputs directly from your multiplier LUT:

```python
import approxtorch as at

# LRE: per-row / per-column linear regression slopes
grad_a, grad_b = at.grad_lut.lre('my_multiplier.txt', qtype='int8', save_path='my_mult')
dx, dw = at.load_lut.load_lre_grad_lut('my_mult_lre_grad_a.txt', 'my_mult_lre_grad_b.txt')
model = at.to_qat_int8(model, lut, grad='lre', dx=dx.cuda(), dw=dw.cuda())
```

For a pair-wise custom gradient, `dx[x + 128, w + 128]` supplies the
derivative with respect to `x`, while `dw[x + 128, w + 128]` supplies the
derivative with respect to `w`:

```python
dx, dw = at.load_lut.load_custom_grad_lut('custom_dx.txt', 'custom_dw.txt')
model = at.to_qat_int8(
    model, lut, grad='custom',
    dx=dx.cuda(), dw=dw.cuda(),
)
```

A smoothing + central-difference method (`at.grad_lut.DATE`) is also included for research comparison.

## API Overview

### Model conversion

```python
at.to_qat_int8(
    model,                    # any nn.Module
    lut,                      # LUT tensor from at.load_lut.load_lut(...)
    x_quantizer='symmetric',  # activation quantizer
    w_quantizer='symmetric',  # weight quantizer
    grad='ste',               # 'ste' | 'lre' | 'custom'
    dx=None, dw=None,         # gradient LUTs, required for 'lre'/'custom'
    conv_only=True,
    ignore_first_conv=True,   # leave the first Conv2d untouched
    scale_momentum=0.05,      # EMA momentum for activation scale updates
    decoupled=True,           # current maintained Conv2d_int8 path
    weight_bits=8,            # 3–8 bit weight quantization (decoupled layer)
)
```

### Layers (`approxtorch.nn`)

Approximate layers can also be used directly, just like their `torch.nn` counterparts:

- `Conv2d_int8` — signed 8-bit approximate convolution
- `Conv2d_uint8` — unsigned 8-bit approximate convolution
- `Conv2d_BN_int8` — convolution with BatchNorm folding
- `Conv2d_gradual_int8` — gradual (blended exact/approximate) convolution for smoother QAT

### Calibration (`approxtorch.calib`)

For static quantization, collect activation/weight scales on a calibration set before converting:

```python
at.calib.calibrate_int8(model, train_loader, num_pictures=1024,
                        save_path='model_calibrated.pth', weight_bits=8)
```

Calibration writes the activation scale and the EMA weight scale into module
buffers, so they are saved together with the model checkpoint.

### Low-level ops (`approxtorch.backend.ops`)

If you want to build your own layers, the raw CUDA ops are exposed as PyTorch custom ops: `bgemm_int8`, `bgemm_uint8`, `gemm_int8`, `gemm_uint8`, `im2col_int8`, `im2col_uint8`, `lut_lookup_int8`, `elementwise_mul`, and the LRE/custom-gradient backward kernels.

## Repository Layout

```text
approxtorch/
├── backend/            # custom op registration + CUDA/C++ kernels
│   └── csrc/
│       ├── cuda/       # approximate (b)gemm, im2col, LUT lookup, backward kernels
│       └── cpu/        # CPU reference implementations
├── nn/                 # approximate Conv2d layers (int8 / uint8 / BN-fused / gradual)
├── convert_model.py    # to_qat_int8 / convert_model — one-line model conversion
├── load_lut.py         # LUT and gradient-LUT loaders
├── grad_lut.py         # gradient LUT generation (LRE, DATE)
├── quant_utils.py      # calibration utilities
└── calib.py            # standalone min-max calibration helper
test/                   # example LUTs, ResNet-20/CIFAR-10 scripts, kernel tests
```

## Citation

If you use ApproxTorch in your work, please cite:

```bibtex
@INPROCEEDINGS{10031519,
  author={Ma, Ke and Kimura, Shinji},
  booktitle={2022 19th International SoC Design Conference (ISOCC)},
  title={ApproxTorch: An Approximate Multiplier Evaluation Environment for CNNs based on Pytorch},
  year={2022},
  pages={77-78},
  doi={10.1109/ISOCC56007.2022.10031519}}
```

## License

[MIT](LICENSE)

## Famous quote

[![Readme Quotes](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark&quote=喝可乐不加冰等于没喝)](https://github.com/piyushsuthar/github-readme-quotes)
