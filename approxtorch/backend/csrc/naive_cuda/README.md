# naive_cuda 算子笔记

## 目录目的

本目录保存 LUT 乘法相关算子的最朴素 CUDA 实现。它们主要用于：

- 作为独立、容易阅读的正确性基准；
- 与 `csrc/cuda/` 和 `csrc/claude/` 中使用 tiling、shared memory、
  split-K、cuBLAS 或 cuDNN 的优化实现进行数值校对；
- 明确固定 LUT 操作数顺序、量化索引方式和 BGEMM 张量布局。

这些 kernel 都采用“一条线程负责一个输出元素，再串行完成归约”的方式，
不以性能为目标。

## 共同约定

### 量化值到 LUT 索引

int8 和 uint8 使用相同的 256 个索引，但映射方式不同：

```text
int8:   index(v) = int(v) + 128     # [-128, 127] -> [0, 255]
uint8:  index(v) = int(v)           # [0, 255]    -> [0, 255]
```

### 二维 LUT 的操作数顺序

所有二维 LUT 均遵守以下不变量：

```text
LUT[A][B]
flat_index = index(A) * 256 + index(B)
```

第一个算子输入是 LUT 的行操作数，第二个算子输入是 LUT 的列操作数。
该顺序对非对称近似乘法器尤其重要，不能随意交换。

### 张量布局

```text
GEMM:
    A [M,K]
    B [K,N]
    C [M,N]

BGEMM:
    X/A [N,K,L]     # N: batch，K: reduction，L: 空间/列
    W/B [O,K]       # O: 输出通道
    Y   [N,O,L]
```

前向 LUT GEMM/BGEMM 输出 `int32`；梯度算子的上游梯度、梯度 LUT 和输出梯度
均为 `float32`。入口会检查 CUDA device、dtype、shape 和 LUT 元素数量，并用
`contiguous()` 规范化非连续输入。kernel 使用 PyTorch 当前 CUDA stream，
不会主动执行全设备同步。

## 算子总览

| 源文件 | Python 算子 | 用途 |
|---|---|---|
| `gemm_naive.cu` | `gemm_int8_naive` | int8 LUT GEMM forward |
| `gemm_naive.cu` | `gemm_uint8_naive` | uint8 LUT GEMM forward |
| `gemm_lre_naive.cu` | `gemm_lre_backward_int8_naive` | int8 LRE GEMM backward |
| `gemm_lre_naive.cu` | `gemm_lre_backward_uint8_naive` | uint8 LRE GEMM backward |
| `gemm_custom_grad_naive.cu` | `gemm_custom_grad_int8_naive` | int8 二维 custom-gradient GEMM backward |
| `gemm_custom_grad_naive.cu` | `gemm_custom_grad_uint8_naive` | uint8 二维 custom-gradient GEMM backward |
| `bgemm_naive.cu` | `bgemm_int8_naive` | int8 LUT BGEMM forward |
| `bgemm_naive.cu` | `bgemm_uint8_naive` | uint8 LUT BGEMM forward |
| `bgeem_lre_navie.cu` | `bgemm_lre_backward_int8_naive` | int8 LRE BGEMM backward |
| `bgeem_lre_navie.cu` | `bgemm_lre_backward_uint8_naive` | uint8 LRE BGEMM backward |
| `bgemm_custom_grad_naive.cu` | `bgemm_custom_grad_int8_naive` | int8 二维 custom-gradient backward |
| `bgemm_custom_grad_naive.cu` | `bgemm_custom_grad_uint8_naive` | uint8 二维 custom-gradient backward |

`bgeem_lre_navie.cu` 的文件名保留了创建时指定的拼写；其中注册的算子名称使用
标准的 `bgemm` 和 `naive` 拼写。

## LUT GEMM forward

文件：`gemm_naive.cu`

输入和输出：

```text
A:   [M,K], int8/uint8
B:   [K,N], int8/uint8
lut: [256,256] 或扁平 [65536], int32
C:   [M,N], int32
```

计算公式：

```text
C[m,n] = sum_k lut[index(A[m,k])][index(B[k,n])]
```

每条线程计算一个 `C[m,n]`，并按 `k=0..K-1` 顺序串行累加。

Python 调用：

```python
C = at.backend.ops.gemm_int8_naive(A, B, lut)
C = at.backend.ops.gemm_uint8_naive(A, B, lut)
```

## LRE GEMM backward

文件：`gemm_lre_naive.cu`

```text
grad_output: [M,N], float32
A:           [M,K], int8/uint8
B:           [K,N], int8/uint8
dx_lut:      [256], float32
dw_lut:      [256], float32

grad_A:      [M,K], float32
grad_B:      [K,N], float32
```

计算公式：

```text
grad_A[m,k] = sum_n grad_output[m,n] * dx_lut[index(B[k,n])]
grad_B[k,n] = sum_m grad_output[m,n] * dw_lut[index(A[m,k])]
```

Python 调用：

```python
grad_A, grad_B = at.backend.ops.gemm_lre_backward_int8_naive(
    grad_output, A, B, dx_lut, dw_lut
)
grad_A, grad_B = at.backend.ops.gemm_lre_backward_uint8_naive(
    grad_output, A, B, dx_lut, dw_lut
)
```

## Custom-gradient GEMM backward

文件：`gemm_custom_grad_naive.cu`

```text
A:       [M,K], int8/uint8
B:       [K,N], int8/uint8
dY:      [M,N], float32
dx_lut:  [256,256] 或扁平 [65536], float32
dw_lut:  [256,256] 或扁平 [65536], float32

grad_A:  [M,K], float32
grad_B:  [K,N], float32
```

计算公式：

```text
grad_A[m,k]
    = sum_n dY[m,n] * dx_lut[index(A[m,k])][index(B[k,n])]

grad_B[k,n]
    = sum_m dY[m,n] * dw_lut[index(A[m,k])][index(B[k,n])]
```

Python 调用：

```python
grad_A, grad_B = at.backend.ops.gemm_custom_grad_int8_naive(
    A, B, dY, dx_lut, dw_lut
)
grad_A, grad_B = at.backend.ops.gemm_custom_grad_uint8_naive(
    A, B, dY, dx_lut, dw_lut
)
```

## LUT BGEMM forward

文件：`bgemm_naive.cu`

输入和输出：

```text
A:   [N,K,L], int8/uint8
B:   [O,K],   int8/uint8
lut: [256,256] 或扁平 [65536], int32
C:   [N,O,L], int32
```

计算公式：

```text
C[n,o,l] = sum_k lut[index(A[n,k,l])][index(B[o,k])]
```

这里明确遵守 `lut[A][B]`，与当前 `nn/bgemm_int8.py` 使用的
`bgemm_fake_*_claude` 数学语义一致。

Python 调用：

```python
C = at.backend.ops.bgemm_int8_naive(A, B, lut)
C = at.backend.ops.bgemm_uint8_naive(A, B, lut)
```

## LRE BGEMM backward

文件：`bgeem_lre_navie.cu`

输入和输出：

```text
grad_output: [N,O,L], float32
x:           [N,K,L], int8/uint8
w:           [O,K],   int8/uint8
dx_lut:      [256],   float32
dw_lut:      [256],   float32

grad_x:      [N,K,L], float32
grad_w:      [O,K],   float32
```

计算公式：

```text
grad_x[n,k,l]
    = sum_o grad_output[n,o,l] * dx_lut[index(w[o,k])]

grad_w[o,k]
    = sum_{n,l} grad_output[n,o,l] * dw_lut[index(x[n,k,l])]
```

`dx_lut` 是用于计算 `grad_x` 的一维斜率表，但它由另一个乘法操作数 `w`
索引；`dw_lut` 用于计算 `grad_w`，由 `x` 索引。这与
`csrc/cuda/bgemm_lre_backward.cu` 和 `grad_lut.lre()` 的约定一致。

Python 调用：

```python
grad_x, grad_w = at.backend.ops.bgemm_lre_backward_int8_naive(
    grad_output, x, w, dx_lut, dw_lut
)

grad_x, grad_w = at.backend.ops.bgemm_lre_backward_uint8_naive(
    grad_output, x, w, dx_lut, dw_lut
)
```

每条 `grad_x` 线程串行遍历 `O`；每条 `grad_w` 线程按 `N`、`L` 顺序串行遍历。

## Custom-gradient BGEMM backward

文件：`bgemm_custom_grad_naive.cu`

输入和输出：

```text
X:       [N,K,L], int8/uint8
W:       [O,K],   int8/uint8
dY:      [N,O,L], float32
dx_lut:  [256,256] 或扁平 [65536], float32
dw_lut:  [256,256] 或扁平 [65536], float32

grad_X:  [N,K,L], float32
grad_W:  [O,K],   float32
```

与 LRE 的一维斜率表不同，custom-gradient 为每一对 `(X,W)` 乘法操作数保存
独立梯度。两个表都严格按照 `[X][W]` 索引：

```text
pair_index = index(X) * 256 + index(W)

grad_X[n,k,l]
    = sum_o dY[n,o,l] * dx_lut[index(X[n,k,l])][index(W[o,k])]

grad_W[o,k]
    = sum_{n,l} dY[n,o,l] * dw_lut[index(X[n,k,l])][index(W[o,k])]
```

Python 调用：

```python
grad_X, grad_W = at.backend.ops.bgemm_custom_grad_int8_naive(
    X, W, dY, dx_lut, dw_lut
)

grad_X, grad_W = at.backend.ops.bgemm_custom_grad_uint8_naive(
    X, W, dY, dx_lut, dw_lut
)
```

每条 `grad_X` 线程串行遍历 `O`；每条 `grad_W` 线程按 `N`、`L` 顺序串行遍历。

## 构建与注册

所有 `.cu` 文件均已列入仓库根目录的 `setup.py`，并通过
`TORCH_LIBRARY_FRAGMENT(approxtorch, m)` 声明 schema、通过
`TORCH_LIBRARY_IMPL(approxtorch, CUDA, m)` 注册 CUDA 实现。

重新构建扩展：

```bash
python setup.py build_ext --inplace
```

也可以使用仓库安装脚本重新进行 editable install：

```bash
./install.sh
```

## 正确性测试

### Forward GEMM/BGEMM

```bash
python approxtorch/backend/csrc/naive_cuda/test_naive_lut_ops.py
```

覆盖 int8/uint8、非对称 LUT、奇数 shape 和非连续输入；GEMM 还会与现有实现
比较，BGEMM 使用独立 oracle 验证。

### LRE backward

```bash
python approxtorch/backend/csrc/naive_cuda/test_bgemm_lre_naive.py
```

使用独立 PyTorch oracle 检查 int8/uint8；int8 还会与现有
`bgemm_lre_backward` 实现比较。

GEMM LRE：

```bash
python approxtorch/backend/csrc/naive_cuda/test_gemm_lre_naive.py
```

### Custom-gradient backward

```bash
python approxtorch/backend/csrc/naive_cuda/test_bgemm_custom_grad_naive.py
```

使用非对称二维梯度 LUT 检查 `[X][W]` 顺序，并分别与优化版
`bgemm_custom_grad_{int8,uint8}_{dx,dw}` 比较。

GEMM custom-gradient：

```bash
python approxtorch/backend/csrc/naive_cuda/test_gemm_custom_grad_naive.py
```

所有测试都需要可用的 CUDA GPU，以及包含本目录源文件的最新编译扩展。

## 实机验证记录

2026-07-13 在以下环境完成实际 CUDA 验证：

```text
GPU:        2 x NVIDIA RTX A6000 (sm_86)
Driver:     570.172.08
CUDA:       12.8
PyTorch:    2.9.1+cu128
```

在 GPU 可见环境重新编译本目录全部 `.cu` 后，上述 5 个测试脚本均完整通过：

```text
PASS forward GEMM/BGEMM:       int8 + uint8
PASS BGEMM LRE backward:       int8 + uint8
PASS BGEMM custom-gradient:    int8 + uint8
PASS GEMM LRE backward:        int8 + uint8
PASS GEMM custom-gradient:     int8 + uint8
```

所有测试均覆盖非连续输入。LRE/custom-gradient 使用非对称 LUT 来检测操作数
顺序错误。梯度测试允许 `rtol=1e-5, atol=1e-5`，用于容纳 CUDA 串行累加和
PyTorch oracle 不同归约顺序造成的正常 float32 舍入差异；实际观察到的最大边界
差异约为 `2.5e-6`。

### 已知的对照实现问题

本机测试时，现有 `bgemm_fake_{int8,uint8}_forward_cuda_claude` forward 对照算子
会长时间卡住，包括非最小 shape。为避免旧优化实现阻塞 naive 基准测试，
`test_naive_lut_ops.py` 中的 BGEMM 只与独立非对称 LUT oracle 比较；GEMM forward
仍会与现有实现比较。

这个问题不影响本目录 naive BGEMM forward 的 oracle 结果，也不影响 BGEMM
LRE/custom-gradient 测试：int8 LRE 已与现有 `bgemm_lre_backward` 比较通过，
int8/uint8 custom-gradient 也已与优化版 `*_dx`、`*_dw` 比较通过。

## 使用限制

- 这些算子仅注册 CUDA backend，没有 CPU 实现。
- 它们是低层 forward/backward primitive，不会自行构建完整 autograd Function。
- forward 使用 `int32` 累加，调用者需要保证 `K` 和 LUT 数值范围不会导致
  期望之外的整数溢出。
- naive 实现仅用于正确性校对和语义参考，不应作为性能基准或生产训练路径。
