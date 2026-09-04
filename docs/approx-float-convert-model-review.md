# Approximate FP16/BF16 与 `convert_model` 代码审查

> 本文是重构前的历史审查快照。问题的当前状态、已完成修改和剩余建议请以
> [approx-float-refactor-final-review.md](approx-float-refactor-final-review.md)
> 为准。

> 审查日期：2026-09-04
> 审查范围：FP16/BF16 LUT 乘法、GEMM/BGEMM、Conv2d、Linear、
> `approxtorch/convert_model.py`、相关测试与文档。
> 本文记录当前实现的问题与建议，不表示这些问题已经修复。

## 1. 结论

FP16/BF16 的底层数值实现是正确的，和代码声明的 RTL 合约一致：

- LUT 操作数顺序为 `LUT[input_fraction][weight_fraction]`；
- sign、exponent、normalization bit 的组合正确；
- naive 与 optimized BGEMM 位级一致；
- Conv2d 的 `unfold -> BGEMM -> reshape -> bias` 几何正确；
- backward 使用精确乘法导数作为 STE，数学关系正确；
- approximate product 使用 FP32 按固定 K 顺序累加，最后只转换一次到
  FP16/BF16。

`convert_model` 在狭义的“把单个普通 `nn.Conv2d` 换成 approximate Conv2d”
场景下也基本正确：它保留了卷积几何、权重、bias、`requires_grad` 和
training/eval 状态，并在遇到不支持的卷积时避免只转换模型的一部分。

但是，它目前还不是一个闭环、安全的 FP16/BF16 整模型转换入口。最需要处理的
问题是模型 dtype 断链、共享模块引用被破坏，以及 qtype-specific 参数混在同一个
接口中。Linear 和公共命名也存在较明显的冗余。

## 2. 问题总览

| 优先级 | 问题 | 影响 |
|---|---|---|
| 高 | FP16/BF16 转换后模型 dtype 不闭合 | 默认配置或 Conv-to-Linear 模型会在 forward 时报错 |
| 高 | 共享 `Conv2d` 引用被拆坏 | 同一层的多个引用转换后不再共享参数和行为 |
| 中 | int8/uint8 LUT 转换期校验不足 | 无效模型能够完成转换，直到第一次 forward 才失败 |
| 中 | `convert_model` 含大量 qtype 无关参数 | 参数被静默忽略，或无意义地触发错误 |
| 中 | Linear 的 `optimized` 当前没有实际作用 | 两条公共路径最终调用完全相同的 CUDA kernel |
| 中 | FP16 legacy `uint32` LUT 被逐层重复转换 | 多卷积模型产生不必要的转换、分配和 LUT 副本 |
| 低 | FP16/BF16 类名和函数名各有两套别名 | 公共 API 面积扩大，文档和自动补全更混乱 |
| 低 | Linear 和 LUT validator 大量重复 | 容易出现行为漂移，当前已经存在 dtype 合约不一致 |
| 中 | 测试、README 和 delivery 文档未同步 | 新 checkout/CI 缺少重要测试，文档描述与实现冲突 |

## 3. 详细问题

### 3.1 FP16/BF16 整模型转换存在 dtype 断链

涉及代码：

- `approxtorch/convert_model.py:32-43`：replacement 权重和 bias 被转换为目标
  FP16/BF16 dtype；
- `approxtorch/convert_model.py:225-230`：只选择 `nn.Conv2d`；
- `approxtorch/nn/_conv2d_float.py:80-87`：严格要求 input、weight 和 bias
  使用目标 dtype。

#### 默认保留首层时

`ignore_first_conv=True` 是默认值。对于一个普通 FP32 模型，第一层保持 FP32，
第二层 approximate Conv2d 已经变成 FP16/BF16。第一层输出直接进入第二层时会
失败：

```text
TypeError: input and weight must have dtype torch.float16,
got torch.float32 and torch.float16
```

#### 模型包含 Linear 时

即使使用 `ignore_first_conv=False` 转换所有卷积，`nn.Linear` 仍保持 FP32。
approximate Conv2d 输出 FP16 后进入 Linear，会失败：

```text
RuntimeError: mat1 and mat2 must have the same dtype,
but got Half and Float
```

当前完整验证脚本之所以能通过，是因为它在转换后额外执行：

```python
model = model.to(device="cuda", dtype=torch.float16)
# 或 dtype=torch.bfloat16
```

这个要求没有进入 `convert_model` 的接口合约，README 的高层转换示例也没有提供
FP16/BF16 用法。

#### 建议

需要先明确高层语义，推荐在以下方案中选择一种：

1. `convert_model` 在浮点 qtype 下同时转换整个模型的浮点参数和 buffer；
2. `convert_model` 仍只替换目标层，但明确返回一个需要随后
   `.to(device, dtype=...)` 的模型，并在文档与错误信息中强制说明；
3. 让 approximate layer 在边界显式 cast input/output，使模型外部保持 FP32。

如果 ApproxTorch 的目标是模拟整网使用 FP16/BF16 activation，方案 1 或 2 更符合
当前 kernel 语义。建议同时提供 `convert_linear` 选项，因为 approximate Linear
已经实现。

### 3.2 共享 `Conv2d` 的引用关系会被破坏

涉及代码：`approxtorch/convert_model.py:225-262`。

`nn.Module.named_modules()` 默认对同一个 module object 去重。例如：

```python
class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        conv = nn.Conv2d(3, 3, 1)
        self.a = conv
        self.b = conv
```

转换前 `model.a is model.b` 为 `True`。当前转换后实测为：

```text
type(model.a) == Conv2d_float16
type(model.b) == Conv2d
model.a is model.b == False
```

这不仅漏掉了一个调用路径，也改变了参数共享语义。

#### 建议

遍历每个父模块的 `parent._modules` 引用，而不是依赖已经去重的
`named_modules()`；同时维护：

```text
id(original_module) -> replacement_module
```

同一个原模块再次出现时，所有引用都应该替换成同一个 replacement。还需要明确
`ignore_first_conv` 是按“唯一模块”还是按“引用出现次数”计算；推荐按唯一模块
计算。

### 3.3 int8/uint8 LUT 没有在转换期完整校验

涉及代码：

- `approxtorch/convert_model.py:183-185`：只检查 LUT 有 65536 个元素；
- `approxtorch/backend/csrc/claude/bgemm_float_claude.cu:80-95`：实际要求
  CUDA、`float32`、65536 个元素，并在后端转 contiguous。

当前可以用错误 dtype 的 LUT 完成模型替换，之后才在 forward 中失败。这与
`convert_model.py:234-235` 所描述的“先构造所有 replacement，保证失败时不部分
修改模型”的目标不完全一致：结构转换成功了，但得到的是一个已被修改且不能运行
的模型。

#### 建议

建立统一的 qtype-specific LUT validator：

- int8/uint8：`float32`、65536 elements；
- FP16：`uint16`、`[1024, 1024]`、contiguous；
- BF16：`uint16`、`[128, 128]`、contiguous；
- device 是否必须在转换期一致，可以继续允许由调用方随后 `.to(device)`，但需要
  在文档中明确。

### 3.4 `convert_model` 的参数混合了四种互斥配置

当前签名位于 `approxtorch/convert_model.py:126-138`：

```python
convert_model(
    model,
    lut,
    qtype="int8",
    grad="ste",
    dx=None,
    dw=None,
    ignore_first_conv=True,
    scale_momentum=0.05,
    update_scale=True,
    weight_bits=8,
    optimized=True,
)
```

各参数的实际适用范围如下：

| 参数 | int8 | uint8 | FP16/BF16 |
|---|---|---|---|
| `grad` | 有效 | 有效 | 只能保持 `ste` |
| `dx`, `dw` | LRE/custom 使用 | LRE/custom 使用 | 禁止使用 |
| `scale_momentum` | 有效 | 有效 | 无效但仍校验 |
| `update_scale` | 有效 | 有效 | 无效但仍校验 |
| `weight_bits` | 3-8 | 只能是默认 8 | 无效但仍校验 |
| `optimized` | 无效但仍校验 | 无效但仍校验 | Conv2d 有效 |

这会产生两类问题：

- 调用方以为参数生效，但它被静默忽略，例如 int8 下的 `optimized=False`；
- 参数根本不参与该 qtype，仍可能导致错误，例如 FP16 下
  `scale_momentum=2`。

#### 建议

短期可以继续保留统一入口，但应只校验当前 qtype 真正使用的参数，并对显式传入的
无关参数报清晰错误或 warning。为了区分“未传入”和“显式传入默认值”，可将
qtype-specific 参数默认值改为 `None`，进入分支后再补默认值。

长期可以使用配置对象减少互斥参数：

```python
Int8Options(grad="ste", weight_bits=8, ...)
UInt8Options(grad="ste", ...)
FloatOptions(kind="fp16", optimized=True, convert_linear=True)
```

### 3.5 Linear 的 `optimized` 当前是无效开关

涉及代码：

- `approxtorch/nn/Linear_float16.py:85-98`；
- `approxtorch/nn/Linear_bfloat16.py:85-98`；
- `approxtorch/backend/csrc/float/gemm_fp16.cu:90-132`；
- `approxtorch/backend/csrc/float/gemm_bf16.cu:90-132`。

Python Linear 根据 `optimized` 在 `gemm_*` 和 `gemm_*_naive` 之间切换，但两条
CUDA 入口最终调用同一个 direct kernel，C++ 中直接使用 `(void)optimized`。

这与 Conv2d 不同：BGEMM 的 `optimized` 会实际选择 tiled 或 naive kernel，所以
Conv2d 上该参数是有效的。

#### 建议

- 如果近期没有独立的 optimized GEMM 计划，从 Linear module 和 functional API
  中弃用 `optimized`；
- 如果保留两个后端名字是为了 reference/debug，应在低层 API 文档中说明它们当前
  位级和执行路径完全相同，不应把开关继续暴露为 Linear 性能配置。

### 3.6 FP16 legacy `uint32` LUT 会被逐层重复转换

涉及代码：

- `approxtorch/convert_model.py:186-196` 允许 FP16 LUT 使用 `uint32`；
- `approxtorch/nn/_conv2d_float.py:252-260` 在每个 Conv2d 构造器中转换为
  `uint16`；
- `approxtorch/float_lut.py:31-50` 的正式 validator 则只接受 `uint16`。

当一个模型有多个卷积并传入 legacy `uint32` LUT 时，每个 replacement 都会独立
执行一次包含 1,048,576 个 entry 的 dtype 转换，并持有各自的新 tensor。正式
LUT loader 已经生成 `uint16`，所以这种兼容逻辑应该集中在入口，而不应散落在每
个 layer 构造器里。

#### 建议

- 在 `convert_model` 开始处只规范化一次 LUT，然后把同一个 `uint16` tensor 传给
  所有 replacement；或者
- 删除 `uint32` 支持，通过明确的 migration helper 处理旧 LUT；
- 让 `float_lut.validate_mantissa_lut`、Conv2d 和 Linear 共用同一合约。

### 3.7 公共命名别名过多

`approxtorch/nn/__init__.py:4-27` 同时公开：

- `Conv2d_float16` 与 `Conv2d_fp16`；
- `Conv2d_bfloat16` 与 `Conv2d_bf16`；
- `conv2d_float16` 与 `conv2d_fp16`；
- `conv2d_bfloat16` 与 `conv2d_bf16`；
- Linear 也有对应的四组重复名字。

这些名称不是不同实现，只是 Python alias。它们扩大了公共 API 面积，也使 README、
类型提示和自动补全需要同时维护两套名字。

#### 建议

选择一套 canonical 命名。由于 qtype 和后端算子已经使用 `fp16`/`bf16`，推荐：

```text
Conv2d_fp16 / Conv2d_bf16
Linear_fp16 / Linear_bf16
conv2d_fp16 / conv2d_bf16
linear_fp16 / linear_bf16
```

较长名字可保留一段兼容期，但从 `__all__` 和主文档移除，并发出 deprecation
warning；不建议立即删除，以免破坏现有用户代码和 checkpoint pickle。

### 3.8 实现重复增加了合约漂移风险

当前至少存在以下重复：

- `Linear_float16.py` 与 `Linear_bfloat16.py` 几乎逐行相同；
- Conv2d、两个 Linear 和 `float_lut.py` 各自维护 LUT validator；
- `bgemm_fp16.py` 与 `bgemm_bf16.py` 的 STE wrapper 高度相似。

Conv2d 已经通过 `_conv2d_float.py` 抽取了共享实现，Linear 应采用相同结构，例如：

```text
approxtorch/nn/_float_common.py
approxtorch/nn/_linear_float.py
```

`_float_common.py` 可以统一维护 dtype、LUT side、entry bits、legacy normalization
和验证逻辑，从根源上消除当前 `uint32` 接受范围不一致的问题。

### 3.9 测试和文档没有被可靠纳入仓库

`.gitignore:11-14` 忽略了 `docs/`、`tools/` 和整个 `tests/`。当前 Git 实际只跟踪
`tests/test_convert_model.py`；以下很有价值的本地验证文件处于 ignored 状态：

- `tests/approx_float_reference.py`；
- `tests/test_float_mantissa_lut.py`；
- `tests/verify_approx_float_ops.py`；
- `tests/verify_approx_float_conv2d.py`；
- `tests/verify_float_mantissa_lut.py`；
- `tools/generate_float_mantissa_lut.py`。

这意味着新 checkout 和 CI 不会得到完整的位级验证，也使新加入的 Linear 没有任何
被跟踪的回归测试。

此外存在两处明显的文档漂移：

1. `docs/approx-float-conv2d-delivery.md` 声称支持 groups/depthwise，并提供
   `groups=2` 示例；当前 `approxtorch/nn/_conv2d_float.py:218-224` 明确拒绝所有
   `groups != 1`。
2. README 的 Model conversion 和 Layers API 仍只列出 int8/uint8，没有说明
   FP16/BF16 的整模型 dtype 要求，也没有列出已经存在的浮点 Linear。

#### 建议

- 不要全局 ignore `tests/`、`tools/` 和 `docs/`；只忽略生成物和大型本地资产；
- 把低层位级验证与 Conv2d 验证接入 CI；
- 为 Linear 添加前向 reference、STE backward、任意 leading dimensions、empty
  tensor、device/dtype contract 和 state_dict 测试；
- 更新 delivery 文档与 README，使 groups 和 dtype 描述与当前实现一致。

## 4. 已确认正确的部分

### 4.1 LUT 乘法语义

`approx_mul_fp16.cuh` 和 `approx_mul_bf16.cuh` 的逻辑与 RTL 合约一致：

1. 从两个 16-bit operand 提取 sign、exponent 和 fraction；
2. 按 `LUT[A_fraction][B_fraction]` 取值，支持非对称 multiplier；
3. LUT entry 的最高有效位作为 exponent normalization；
4. sign 使用 XOR；
5. exponent 在 FP16 的 5 bit 或 BF16 的 8 bit 字段内回绕；
6. 任一 operand 为正零或负零时，结果强制为正零。

这里模拟的是 RTL，不是完整 IEEE-754。NaN、Inf、overflow、underflow、subnormal
normalization 和乘法 rounding 都没有特殊处理。只要目标硬件采用同样语义，这不是
实现错误；如果目标是替代标准 IEEE FP16/BF16 乘法，则需要重新定义特殊值和指数
边界行为。

### 4.2 GEMM/BGEMM

- approximate product 先打包为目标 16-bit dtype；
- 每个 product 再转成 FP32；
- accumulator 按固定 K 顺序使用 FP32 加法；
- dot product 结束后只做一次 FP16/BF16 round-to-nearest-even 转换；
- optimized BGEMM 保持与 naive 相同的 reduction 顺序，因此可做到位级一致；
- 使用当前 CUDA stream，并正确设置 device guard；
- empty output、`K=0` 和不规则 tile/tail 均有正确处理。

### 4.3 Conv2d 与 backward

- `F.unfold` 的 K 顺序和 BGEMM 的 weight flatten 顺序匹配；
- stride、zero padding 和 dilation 的输出尺寸计算正确；
- input 是 LUT row operand，weight 是 column operand；
- bias 在最终 16-bit output 上添加；
- BGEMM backward 使用普通乘法的矩阵梯度，`F.unfold` 的 autograd 再负责 fold 回
  input，构成标准 Conv2d STE；
- `requires_grad`、bias 有无和 eval/training 状态在转换时被正确保留。

当前明确限制为 CUDA、`groups=1`、zero padding，不支持 string padding。

## 5. 实际验证结果

验证环境：

```text
GPU: 2 x NVIDIA RTX 6000 Ada Generation, compute capability 8.9
PyTorch: 2.9.1+cu128
CUDA available: True
```

执行过以下验证：

```bash
python tests/test_convert_model.py -v
python tests/verify_approx_float_ops.py \
    --lut-dir /tmp/approxtorch-review-luts.bpCQug
python tests/verify_approx_float_conv2d.py \
    --lut-dir /tmp/approxtorch-review-luts.bpCQug
python -m compileall -q approxtorch tests tools
```

结果：

- `test_convert_model.py`：9/9 通过；
- FP16/BF16 exhaustive/boundary/random multiplier 验证通过；
- exact 与非对称 LUT 的 GEMM/BGEMM 严格 reference 验证通过；
- optimized/naive BGEMM 位级一致；
- FP32 accumulator、empty/K=0、device、stream 和 contract 验证通过；
- FP16/BF16 Conv2d 严格 forward、STE backward、module/state/device 验证通过；
- 手动补测 FP16/BF16 Linear 的三维 input、bias 和 STE backward 通过；
- Python 语法编译通过。

环境中没有安装 pytest，因此 `test_float_mantissa_lut.py` 没有通过 pytest runner
执行；其覆盖的核心 LUT 内容和 kernel 行为已经由独立验证脚本覆盖。

另外通过最小模型稳定复现了：

- 默认 `ignore_first_conv=True` 的 FP32 -> FP16 Conv dtype 错误；
- converted FP16 Conv -> untouched FP32 Linear 的 dtype 错误；
- shared Conv 只替换一个引用并失去 sharing。

## 6. 推荐修复顺序

### 第一阶段：修复模型语义

1. 明确 FP16/BF16 模型的 dtype policy，并让接口、README 和 runtime 行为一致；
2. 决定 `convert_model` 是否转换 `nn.Linear`，建议提供显式选项；
3. 修复共享 module 的遍历与 replacement cache；
4. 添加以上三类回归测试。

### 第二阶段：收紧合约

1. 集中实现 qtype-specific LUT validator；
2. 在转换前一次性规范化 FP16 legacy `uint32` LUT；
3. 只校验当前 qtype 使用的参数；
4. 对被忽略的参数报 warning/error，而不是静默接受。

### 第三阶段：清理接口与重复实现

1. 决定 Linear 是否需要真正的 optimized GEMM；
2. 选择 `fp16`/`bf16` 作为 canonical 公共命名，逐步弃用长别名；
3. 抽取 `_float_common.py` 和 `_linear_float.py`；
4. 更新 README、delivery 文档和 API 列表；
5. 调整 `.gitignore` 并把完整验证脚本纳入版本控制和 CI。

## 7. 建议增加的回归测试

```text
convert_model_fp16_default_dtype_policy
convert_model_bf16_default_dtype_policy
convert_model_float_conv_to_linear
convert_model_preserves_shared_conv_aliases
convert_model_rejects_wrong_integer_lut_dtype_before_mutation
convert_model_rejects_or_warns_on_irrelevant_options
convert_model_normalizes_legacy_fp16_lut_once
linear_fp16_forward_backward_and_empty_shapes
linear_bf16_forward_backward_and_empty_shapes
linear_optimized_flag_has_distinct_behavior_or_is_removed
```

这些测试应同时检查模型结构、参数 object sharing、dtype/device、forward 和
backward，而不只检查 replacement class 名称。
