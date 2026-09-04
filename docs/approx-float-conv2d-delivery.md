# 近似 FP16/BF16 Conv2d

## 1. 实现范围

`approxtorch.nn.conv2d_fp16`、`conv2d_bf16`、
`Conv2d_fp16` 和 `Conv2d_bf16` 复用 PyTorch CUDA `unfold` 生成
`X[N,K,L]`，再调用对应的 LUT BGEMM。当前支持普通卷积（仅 `groups=1`）、stride、zero padding、dilation、可选 bias、naive/optimized BGEMM 和 STE
backward。

## 2. Forward 语义

对每个输出：

1. `unfold` 按 input channel、kernel row、kernel column生成 K 维。
2. 每个 tap 使用 `LUT[input_fraction][weight_fraction]`，顺序不可交换。
3. LUT entry 直接给出乘积 fraction 和 normalization bit；exponent 按字段宽度回绕。
4. 唯一乘法特殊规则是 `+0/-0 * 任意值 = +0`。
5. 每个 FP16/BF16 LUT 乘积转换到 FP32 register，按固定 K 顺序累加。
6. 完成 dot product 后只转换一次到目标 FP16/BF16，再以该 dtype 加 bias。

乘法不检查 NaN、Inf、subnormal、overflow、underflow 或 rounding。padding 由
`unfold` 注入正零，按照零传播规则，其与任何 weight 相乘都得到正零。
零传播使用无分支位掩码，不会产生数据相关的 warp divergence。

## 3. LUT 合约

| qtype | shape | dtype | entry layout |
|---|---:|---|---|
| FP16 | `[1024,1024]` | `torch.uint32` | bit 10 normalization，bits 9:0 fraction |
| BF16 | `[128,128]` | `torch.uint32` | bit 7 normalization，bits 6:0 fraction |

LUT 必须 contiguous、row-major，并与 input/weight 位于同一 CUDA device。

```bash
python tools/generate_float_mantissa_lut.py   --kind all --format all   --output-dir /tmp/approxtorch-luts
```

```python
import approxtorch as at

lut16 = at.float_lut.load_exact_lut(
    "fp16", "cuda:0", directory="/tmp/approxtorch-luts")
lutb = at.float_lut.load_exact_lut(
    "bf16", "cuda:0", directory="/tmp/approxtorch-luts")
```

## 4. 调用示例

```python
import torch
import approxtorch as at

x = torch.randn(8, 16, 32, 32, device="cuda", dtype=torch.float16)
w = torch.randn(32, 16, 3, 3, device="cuda", dtype=torch.float16)
b = torch.randn(32, device="cuda", dtype=torch.float16)

y = at.nn.conv2d_fp16(
    x, w, lut16, b,
    stride=1, padding=1, dilation=1, groups=1,
    optimized=True,
)
```

Module API 的 LUT 是 registered buffer，会进入 `state_dict` 并随
`.cuda()` / `.to(device)` 移动。`convert_float_model(..., qtype="fp16" | "bf16")`
会替换 `nn.Conv2d` 与 `nn.Linear`，保留 geometry、bias、training 状态和
`requires_grad`，并把 weight/bias 保存为对应的 FP16 或 BF16 `nn.Parameter`。

## 5. Backward

LUT 是离散函数，forward 没有普通解析导数。STE 路径使用普通
`conv2d_input` / `conv2d_weight` 作为 surrogate gradient；bias 使用普通
autograd，LUT 不求梯度。因此 backward 表示“把近似 forward 当成普通乘法卷积”
的梯度近似，并不是 RTL multiplier 的真实导数。

## 6. 合约与限制

- forward 仅支持 CUDA。
- input、weight、bias 必须匹配目标 dtype/device。
- LUT 必须符合对应 shape、`torch.uint32`、contiguous 和 device 合约。
- 支持 int 或二元 tuple 的 stride/padding/dilation。
- 只支持 zero padding。
- 当前仅支持 `groups=1`，grouped 和 depthwise 会明确报错。
- native `torch.conv2d` 使用不同乘法和累加语义，不能作为逐位 reference。

## 7. 验证

```bash
python tests/verify_approx_float_conv2d.py   --lut-dir /tmp/approxtorch-luts
```

FP16/BF16 均已覆盖 exact/非对称 LUT、普通卷积、stride、
padding、dilation、bias、raw 16-bit 边界、FP32 累加和单次最终转换、
optimized/naive 逐位一致、
module/state/device、STE、模型转换、empty/channels-last、错误检查和
non-default CUDA stream。
