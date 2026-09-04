# Approximate FP16/BF16 GEMM/BGEMM

## 1. 实现范围

基于 LUT 的 FP16/BF16 乘法已经拆成四个独立 CUDA 编译单元：

- `approxtorch/backend/csrc/float/gemm_fp16.cu`
- `approxtorch/backend/csrc/float/gemm_bf16.cu`
- `approxtorch/backend/csrc/float/bgemm_fp16.cu`
- `approxtorch/backend/csrc/float/bgemm_bf16.cu`

公共参数检查和 launch 辅助函数位于
`approxtorch/backend/csrc/float/approx_float_cuda_common.cuh`。旧的
`approxtorch/backend/csrc/cuda/approx_float_gemm.cu` 和 fast multiplier 文件已删除，
`setup.py` 直接编译上述四个文件。

对外算子保持不变：

- `gemm_fp16` / `gemm_fp16_naive`
- `gemm_bf16` / `gemm_bf16_naive`
- `bgemm_fp16` / `bgemm_fp16_naive`
- `bgemm_bf16` / `bgemm_bf16_naive`
- `approx_mul_fp16` / `approx_mul_bf16`

## 2. 数值语义

乘法仿真目标是当前 RTL/Python 模型，不是完整 IEEE-754：

1. 从两个 16-bit operand 提取 sign、exponent 和 fraction。
2. 严格按 `LUT[left_fraction][right_fraction]` 读取 entry；操作数顺序不可交换。
3. LUT entry 给出结果 fraction 和 exponent normalization bit。
4. 结果 sign 为 XOR；exponent 执行 bias 修正后按字段宽度回绕。
5. 唯一特殊规则是 `+0/-0 * 任意值 = +0`。
6. NaN、Inf、subnormal、overflow、underflow 和 rounding 均不做特殊检测。

零传播在 multiplier 中用最终位掩码实现，没有数据相关的 `if/return`，因此不会因
零值在 warp 内分布不同而引入分支 divergence。LUT 使用 `__ldg` 进入只读缓存路径。

GEMM/BGEMM 的每个 LUT 乘积仍先打包成 FP16/BF16，然后转换为 FP32。
accumulator 是 FP32 register，并按 `k = 0, 1, ..., K-1` 顺序逐次调用
`__fadd_rn`。完整 dot product 结束后只调用一次 `__float2half_rn` 或
`__float2bfloat16_rn`。没有 FMA、Tensor Core、cuBLAS、split-K 或 reduction
reorder；这保证 optimized 与 naive 逐位一致。

## 3. LUT 合约

| kind | shape | dtype | entry layout | 大小 |
|---|---:|---|---|---:|
| FP16 | `[1024,1024]` | `torch.uint32` | bit 10 normalization，bits 9:0 fraction | 4,194,304 bytes |
| BF16 | `[128,128]` | `torch.uint32` | bit 7 normalization，bits 6:0 fraction | 65,536 bytes |

两种 LUT 都必须是 CUDA、contiguous、row-major，并与输入处于同一 device。
FP16 和 BF16 LUT 统一使用 `uint32` 存储；虽然有效 payload 分别只有 11 bit 和
8 bit，但 Python 校验、序列化格式和 CUDA kernel 指针类型保持单一、明确的 ABI。

exact LUT 使用纯整数生成：

```bash
python tools/generate_float_mantissa_lut.py   --kind all --format all   --output-dir /tmp/approxtorch-validation-luts
```

## 4. Kernel 设计

### GEMM

`A[M,K] @ B[K,N] -> C[M,N]` 使用 256 threads/block 的
one-thread-per-output 映射和 grid-stride loop。同一 warp：

- 连续读取 B 的相邻 column，global load coalesced；
- 读取相同 A scalar，可利用 warp broadcast/cache；
- 访问相同 LUT row，增强 read-only cache locality；
- K loop 使用 `#pragma unroll 4`。

shared-memory 和多 accumulator GEMM 版本均做过实测，但严格的逐 K FP32
dependency chain 使同步和额外寄存器开销大于复用收益。因此当前
`gemm_*` 与 `gemm_*_naive` 都分派到测得最快的 direct kernel；保留两个 API
用于兼容及逐位对照。

### BGEMM

`X[N,K,L]` 与 `W[O,K]` 输出 `Y[N,O,L]`。

naive 路径是一线程一个输出。optimized 路径使用 256 threads/block，也就是
8 个 warp：

- 一个 warp 固定一个 L 位置，lane 跨相邻 O；
- 同一 warp 共享 X operand 和 LUT row；
- X/W cooperative load 到 shared memory；
- W 从 `[O,K]` 转置为按 K 访问的 shared layout；
- shared W stride 加 2 个元素 padding，消除同 K、跨 O 读取的 bank conflict；
- 每个 lane 按 O 大小持有 1、2 或 4 个独立 accumulator；
- FP16 使用 K tile 32，BF16 使用 K tile 64；
- tail K/O/L 均受边界保护，累加顺序不变。

实测 dispatch：

| kind | tiled 条件 |
|---|---|
| FP16 | `K >= 32 && L >= 8 && O >= 16` |
| BF16 | `K >= 64 && L >= 192 && O >= 16` |

较小或 BF16 的短 L shape 使用 direct kernel，避免 shared-memory 同步和尾块成本。

## 5. CUDA/PyTorch 集成

所有入口都检查 CUDA、dtype、维度、contiguous、inner K、LUT shape/dtype 和
同 device。launch 使用：

- `at::cuda::OptionalCUDAGuard`
- `at::cuda::getCurrentCUDAStream()`
- `C10_CUDA_KERNEL_LAUNCH_CHECK()`
- 64-bit shape/index arithmetic
- 安全的 grid-size 检查

`K=0` 时非空输出由 kernel 的零 accumulator 产生；空输出直接返回。

## 6. 验证

最终实现已经通过：

```bash
python setup.py build_ext --inplace
python tests/verify_float_mantissa_lut.py
python tests/verify_approx_float_ops.py   --lut-dir /tmp/approxtorch-validation-luts
python tests/verify_approx_float_conv2d.py   --lut-dir /tmp/approxtorch-validation-luts
```

覆盖内容包括：

- FP16 1,048,576 个 LUT entry 和 BF16 16,384 个 entry exhaustive 验证；
- 每种 dtype 的所有 65,536 个 operand bit pattern 与边界 operand 组合；
- 每种 dtype 额外 1,000,000 个随机乘法 pair；
- exact LUT 和非对称 LUT 下 GEMM/BGEMM 与独立 FP32 累加 reference 逐位一致；
- 专门的累加精度回归验证只在 dot product 末尾转换一次到 FP16/BF16；
- empty、`K=0`、不规则尺寸、tile/tail 边界；
- dtype/shape/contiguous/device 错误；
- non-default current CUDA stream；
- Conv2d、Linear、STE 和模型转换回归。

环境中没有安装 pytest，因此使用可独立运行的完整验证脚本。

## 7. 性能结果

RTX 6000 Ada 上代表性 benchmark（CUDA event，20 repeats）：

> 注意：下表是旧 uint16 LUT 存储条件下的历史数据。当前 uint32 ABI 已完成
> 正确性验证，但尚未重跑性能基准；发布或比较性能前需要更新本节。

| dtype | case | naive ms | optimized ms | speedup |
|---|---|---:|---:|---:|
| FP16 | BGEMM `16x256x196x128` | 1.1563 | 0.3252 | 3.555x |
| FP16 | BGEMM `32x576x1024x64` | 12.1580 | 5.0812 | 2.393x |
| FP16 | irregular BGEMM | 0.3475 | 0.1568 | 2.216x |
| BF16 | BGEMM `16x256x196x128` | 0.3569 | 0.2494 | 1.431x |
| BF16 | BGEMM `32x576x1024x64` | 3.3474 | 1.0369 | 3.228x |
| BF16 | irregular BGEMM | 0.1001 | 0.1003 | 0.998x |

大型 case 的 optimized/naive 几何平均为 FP16 1.691x、BF16 1.430x。
GEMM 的 optimized/naive 入口使用同一个最快 direct kernel，计时差异应视为
GPU clock/cache 浮动。

复现命令：

```bash
python benchmarks/benchmark_approx_float_gemm.py   --lut-dir /tmp/approxtorch-validation-luts   --warmup 3 --repeats 20   --json /tmp/approxtorch-final-benchmark.json
```

## 8. 限制

- 仅 CUDA forward；没有自定义 backward。
- 只接受 contiguous tensor，不隐式复制。
- LUT 乘法故意不实现 IEEE-754 exception/rounding 语义。
- 优化阈值基于 RTX 6000 Ada；其他架构建议重新 benchmark。
- 严格的 LUT 乘积、FP32 逐 K 累加和单次最终转换限制了会重排 reduction 的优化。
