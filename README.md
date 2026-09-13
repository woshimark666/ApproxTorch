# ApproxTorch

**ApproxTorch** is a PyTorch extension for simulating **approximate multipliers** inside Convolutional Neural Networks — for both **training (QAT)** and **inference** — entirely on the GPU.

Approximate multipliers are hardware circuits that trade a small amount of arithmetic accuracy for large savings in power and area. Before taping out such hardware, you need to know how much CNN accuracy you lose — and how much you can recover by retraining. ApproxTorch answers this question by replacing every multiplication inside convolution layers with a **look-up table (LUT)** of your multiplier's actual behavior, computed by highly optimized custom CUDA kernels.

Because the simulation is purely LUT-driven, **any 8-bit (signed or unsigned)
approximate multiplier can be simulated**. ApproxTorch also supports normalized-
mantissa LUTs for approximate **FP16 and BF16** GEMM and BGEMM.

```text
 FP32 input ──► quantize (int8) ──► im2col ──► LUT-based approximate GEMM ──► dequantize ──► FP32 output
                                                  (custom CUDA kernel)
```

## Features

- 🚀 **GPU-accelerated**: hand-written CUDA kernels for LUT-based approximate batched GEMM — fast enough to retrain networks like ResNet on ImageNet-scale data.
- 🎯 **Train *and* infer**: full autograd support, so you can do approximate-multiplier-aware retraining (QAT), not just evaluation.
- 🧮 **Any 8-bit multiplier**: behavior is defined entirely by a 256×256 LUT text file.
- 🧠 **Approximate FP16/BF16**: LUT mantissa GEMM/BGEMM with ordered FP32
  accumulation and one final 16-bit conversion.
- 🔁 **Gradient estimators** for backpropagating through the non-differentiable LUT:
  - **STE** — straight-through estimator (default)
  - **LRE** — linear-regression-estimated gradient LUTs
  - **Custom** — one derivative for every quantized `(x, w)` pair
- ⚖️ **EMA quantization**: per-tensor activation scale and per-channel weight scale are EMA-updated during training, with arbitrary weight bit-width (**3–8 bit**) for int8 weights.
- 🔌 **Separated model conversion**: `at.convert_int_model(...)` handles integer Conv2d; `at.convert_float_model(...)` handles FP16/BF16 Conv2d and Linear.
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
model = at.convert_int_model(
    model, lut,
    qtype='int8',             # 'int8' | 'uint8'
    grad='ste',               # gradient estimator: 'ste' | 'lre' | 'custom'
    ignore_first_conv=True,   # keep the first conv exact (common QAT practice)
    weight_bits=8,            # 3–8 for int8; uint8 is fixed at 8
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
model = at.convert_int_model(model, lut, grad='lre', dx=dx.cuda(), dw=dw.cuda())
```

For a pair-wise custom gradient, `dx[x + 128, w + 128]` supplies the
derivative with respect to `x`, while `dw[x + 128, w + 128]` supplies the
derivative with respect to `w`:

```python
dx, dw = at.load_lut.load_custom_grad_lut('custom_dx.txt', 'custom_dw.txt')
model = at.convert_int_model(
    model, lut, grad='custom',
    dx=dx.cuda(), dw=dw.cuda(),
)
```

A smoothing + central-difference method (`at.grad_lut.DATE`) is also included for research comparison.

### BF16 mantissa gradient LUTs

BF16 layers support a custom CUDA backward using two precomputed, fixed FP32
gradient LUTs. Each tensor must be contiguous and one-dimensional with shape
`[16384]`; both stay on the same CUDA device as the operands. Prepare and upload
them once before training:

```python
device = torch.device("cuda")
grad_x_lut = (
    torch.load("bf16_grad_x.pt", map_location="cpu", weights_only=True)
    .detach().float().reshape(-1).contiguous().to(device)
)
grad_w_lut = (
    torch.load("bf16_grad_w.pt", map_location="cpu", weights_only=True)
    .detach().float().reshape(-1).contiguous().to(device)
)
lutb = at.load_lut.load_float_lut("my_bf16_lut.pt", qtype="bf16").to(device)
model = at.convert_float_model(
    model, lutb, qtype="bf16", ignore_first_conv=False,
    grad_x_lut=grad_x_lut, grad_w_lut=grad_w_lut,
).to(device=device, dtype=torch.bfloat16)
```

Supplying both gradient LUTs selects custom gradients in BF16 Conv2d and Linear.
Omitting both retains the existing STE. The same keyword arguments are accepted
by `Conv2d_bf16`, `Linear_bf16`, `conv2d_bf16`, and `linear_bf16`; FP16 keeps its
existing interface. Modules register the tables as buffers, preserve their FP32
dtype when casting the model, and share already resident tensors. The forward
mantissa LUT remains a separate uint32 tensor with shape `[128, 128]`, and the
forward algorithm is unchanged.

The flattened gradient index always keeps the input fraction first:

```cpp
index = ((bits_x & 0x7F) << 7) | (bits_w & 0x7F);
```

For normal BF16 operands, table entries are `df/dm_x` and `df/dm_w` for the
approximate mantissa product `f(m_x, m_w)`. CUDA propagates

\[
g_x = g_{out}\,s_w\,2^{E_w-127}\,\mathrm{grad\_x\_lut}[index],\qquad
g_w = g_{out}\,s_x\,2^{E_x-127}\,\mathrm{grad\_w\_lut}[index].
\]

Intermediate multiplication and reduction use FP32; returned gradients keep
the BF16 operand shapes and dtype. GEMM and BGEMM sum over the existing
contraction dimensions, including all batch and spatial positions for shared
weights. Kernels only read the supplied gradient tables and do not generate or
update them. This custom backward supports first-order gradients only.

Special values follow the existing RTL convention: either signed-zero operand
masks both propagated gradients to zero. Nonzero operands with exponent fields
0 or 255 retain their raw fraction and use `E - 127`; no IEEE special-value
propagation or flush-to-zero rule is added. The forward's eight-bit exponent
wraparound remains unchanged. The exact-product gradient identity is tested for
normal operands, for which `grad_x_lut[index] = 1 + F_w/128` and
`grad_w_lut[index] = 1 + F_x/128`.

Run the correctness tests after building the extension:

```bash
python -m pytest approxtorch/backend/csrc/float/test_bf16_custom_grad.py -q
```

## API Overview

### Model conversion

```python
at.convert_int_model(
    model,                    # any nn.Module
    lut,                      # LUT tensor from at.load_lut.load_lut(...)
    qtype='int8',             # 'int8' | 'uint8'
    grad='ste',               # 'ste' | 'lre' | 'custom'
    dx=None, dw=None,         # gradient LUTs, required for 'lre'/'custom'
    ignore_first_conv=True,   # leave the first Conv2d untouched
    scale_momentum=0.05,      # EMA momentum for quantization statistics
    update_scale=True,        # update statistics while training
    weight_bits=8,            # 3–8 for int8; uint8 is fixed at 8
)
```

Floating-point conversion has its own API and converts both Conv2d and Linear:

```python
# Generate these files once. Both FP16 and BF16 LUT tensors are uint32.
# python tools/generate_float_mantissa_lut.py --kind all --format all \
#     --output-dir /tmp/approxtorch-luts
lut16 = at.float_lut.load_exact_lut(
    "fp16", "cuda:0", directory="/tmp/approxtorch-luts"
)
model = at.convert_float_model(
    model,
    lut16,
    qtype="fp16",             # "fp16" | "bf16"
    ignore_first_conv=True,   # affects Conv2d only; every Linear is converted
    optimized=True,
).to(device="cuda", dtype=torch.float16)
```

For arbitrary approximate mantissa LUT files, the general loader accepts both
short and long dtype names and returns a contiguous CPU uint32 tensor:

```python
lut16 = at.load_lut.load_lut("my_fp16_lut.bin", qtype="float16").cuda()
lutb = at.load_lut.load_float_lut("my_bf16_lut.pt", qtype="bf16").cuda()
```

Supported inputs are whitespace-delimited text, raw little-endian uint32
`.bin`, and PyTorch `.pt`/`.pth` tensors. Use `float_lut.load_exact_lut` when
loading generated exact LUTs with manifest/hash verification.

The converted Conv2d and Linear weights and biases are nn.Parameter tensors in
torch.float16 or torch.bfloat16, respectively. The converter does not cast
unreplaced layers automatically, so cast the surrounding model and its inputs
to a compatible dtype as shown above.

### Layers (`approxtorch.nn`)

Approximate layers can also be used directly, just like their `torch.nn` counterparts:

- `Conv2d_int8` — signed 8-bit approximate convolution
- `Conv2d_uint8` — unsigned 8-bit approximate convolution
- `Conv2d_fp16` / `Conv2d_bf16` — fixed-dtype approximate convolution
- `Linear_fp16` / `Linear_bf16` — fixed-dtype approximate linear layer

### Calibration (`approxtorch.calib`)

For static quantization, collect activation/weight scales on a calibration set before converting:

```python
at.calib.calibrate_int8(model, train_loader, num_pictures=1024,
                        save_path='model_calibrated.pth', weight_bits=8)
```

Calibration writes the activation scale and the EMA weight scale into module
buffers, so they can be loaded after model conversion.

### Low-level ops (`approxtorch.backend.ops`)

If you want to build your own layers, the raw CUDA ops are exposed as PyTorch
custom ops, including the integer operations and `approx_mul_fp16`,
`approx_mul_bf16`, `gemm_fp16`, `gemm_bf16`, `bgemm_fp16`, and `bgemm_bf16`
(plus their `_naive` reference kernels).

The matching STE autograd wrappers are split by dtype:

```python
from approxtorch.nn import bgemm_fp16, bgemm_bf16

y16 = bgemm_fp16.bgemm_fp16_ste(x16, w16, lut16)
yb = bgemm_bf16.bgemm_bf16_ste(xb, wb, lutb)
```

BF16 custom-gradient wrappers use the same forward operators and add the two
flat FP32 gradient tables:

```python
from approxtorch.nn import (
    approx_mul_bf16_custom, gemm_bf16_custom, bgemm_bf16_custom,
)

# Elementwise: x and w have the same shape.
y = approx_mul_bf16_custom(x, w, lutb, grad_x_lut, grad_w_lut)
# GEMM: A [M,K], B [K,O] -> [M,O].
y = gemm_bf16_custom(A, B, lutb, grad_x_lut, grad_w_lut)
# BGEMM: X [N,K,L], W [O,K] -> [N,O,L].
y = bgemm_bf16_custom(X, W, lutb, grad_x_lut, grad_w_lut)
```

## Repository Layout

```text
approxtorch/
├── backend/            # custom op registration + CUDA/C++ kernels
│   └── csrc/
│       ├── cuda/       # approximate (b)gemm, im2col, LUT lookup, backward kernels
│       └── cpu/        # CPU reference implementations
├── nn/                 # approximate integer and FP16/BF16 Conv2d/Linear layers
├── convert_model.py    # split integer and FP16/BF16 model conversion
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
