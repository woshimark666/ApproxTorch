# Approximate FP16/BF16 重构与最终审查报告

> 审查日期：2026-09-04
> 范围：浮点 Conv2d/Linear、mantissa LUT、CUDA GEMM/BGEMM、模型转换、测试与公开文档。

## 结论

本轮要求已经落地。FP16 与 BF16 模块现在有明确分开的类和 repr；
Conv2d/Linear 的 weight、bias 都保存为对应 dtype 的 nn.Parameter；
两种 mantissa LUT 从生成、加载、Python 校验到 CUDA 指针统一为 uint32；
模型转换拆成整数与浮点两个入口，浮点入口同时转换 Conv2d 和 Linear。

在当前明确支持的范围内，没有发现阻断性的数值或状态复制错误。仍有几项
能力限制与兼容性/冗余问题，见“剩余问题与建议”。

## 已完成的改动

| 项目 | 当前行为 |
|---|---|
| FP16 Conv2d | 继承 ApproxConv2dFloat16，参数为 torch.float16，repr 明示 dtype=torch.float16 |
| BF16 Conv2d | 继承 ApproxConv2dBFloat16，参数为 torch.bfloat16，repr 明示 dtype=torch.bfloat16 |
| FP16 Linear | weight/bias 为 torch.float16 nn.Parameter，repr 明示 torch.float16 |
| BF16 Linear | weight/bias 为 torch.bfloat16 nn.Parameter，repr 明示 torch.bfloat16 |
| 浮点 LUT | FP16 [1024,1024]、BF16 [128,128]，两者都必须为 contiguous torch.uint32 |
| CUDA ABI | 所有浮点 multiply/GEMM/BGEMM kernel 均读取 const uint32_t* LUT |
| 整数转换 | convert_int_model，仅接受 int8/uint8 及整数域参数 |
| 浮点转换 | convert_float_model，仅接受 fp16/bf16，同时转换 nn.Conv2d 与 nn.Linear |
| 图结构 | 保留同一模块对象的多属性共享关系；转换失败前不修改原模型 |

## LUT 文件格式

- 二进制 entry 固定为 4 bytes little-endian。
- FP16 文件数据区大小为 4,194,304 bytes。
- BF16 文件数据区大小为 65,536 bytes。
- manifest 已升级到 schema_version=2、version=rtl-mantissa-result-v2。
- loader 同时检查版本、dtype、element size、总大小、byte order、shape 和 SHA-256。
- 旧 uint16/v1 LUT 会被拒绝，需要重新生成，避免静默解释错误。

重新生成命令：

```bash
python tools/generate_float_mantissa_lut.py \
  --kind all --format all --output-dir /tmp/approxtorch-luts
```

## convert API 合约

整数模型：

```python
model = approxtorch.convert_int_model(
    model, lut, qtype="int8", grad="ste", ignore_first_conv=True
)
```

浮点模型：

```python
model = approxtorch.convert_float_model(
    model, lut16, qtype="fp16", ignore_first_conv=True, optimized=True
)
model = model.to(device="cuda", dtype=torch.float16)
```

convert_float_model 的 ignore_first_conv 只跳过第一个唯一的 Conv2d；Linear
始终转换。转换会保留 weight/bias 数值（转换到目标 dtype）、requires_grad、
training 状态、卷积 geometry、bias 有无以及共享模块 alias。

旧的可调用函数 convert_model 已移除。Python 导入机制仍可能让
approxtorch.convert_model 指向同名子模块，但它不是公开的转换函数。

## 验证结果

本轮实际执行并通过：

```text
python setup.py build_ext --inplace
python tests/test_convert_model.py -v                         12/12 PASS
python tests/verify_float_mantissa_lut.py                     PASS
python tests/verify_approx_float_ops.py --lut-dir <v2-luts>   PASS
python tests/verify_approx_float_conv2d.py --lut-dir <v2-luts> PASS
git diff --check                                              PASS
python -m compileall ...                                      PASS
```

CUDA 验证覆盖：exact 与非对称 LUT、全部 65,536 种 16-bit operand pattern
配合边界 operand、额外 1M 随机乘法、严格独立 reference、FP32 固定顺序累加、
empty/K=0/tile tail、current stream、device/dtype/layout 错误、Conv2d 与
Linear module/functional forward，以及 STE backward。环境包含两张 RTX 6000 Ada，
也覆盖了多 GPU LUT cache/device 行为。

## 剩余问题与建议

### P1：浮点 Conv2d 仍只支持 groups=1

groups/depthwise 会明确抛出 NotImplementedError，convert_float_model 会在任何
替换发生前失败，因此不会留下半转换模型。这不是静默错误，但会阻止直接转换
包含 depthwise convolution 的网络。建议下一阶段实现按 group 分块的 BGEMM，
并补回 grouped/depthwise 的严格 reference 测试。

### P1：converter 不自动统一整个模型的 dtype

转换层的参数 dtype 正确，但未替换的 BatchNorm、首个 Conv2d、常量 buffer 和
调用方输入不会由 convert_float_model 自动处理。直接把 FP32 input 送入 FP16
替换层会报 dtype mismatch。当前选择是显式且可控的：调用方在转换后执行
model.to(dtype=...)，并确保 input dtype 一致。若要自动处理，建议新增独立的
cast_model 参数或包装 API，不要把隐式全模型转换塞回 converter。

### P2：Linear 的 optimized 参数目前没有实际分支收益

Linear 的 optimized=True/False 分别调用 gemm_* 与 gemm_*_naive，但当前 CUDA
实现中两者最终使用同一个 direct kernel。该参数对 Linear 是兼容性接口而不是
性能开关。建议二选一：实现真正独立的 optimized GEMM，或在下一个 breaking
release 从 Linear/linear_* 中删除 optimized；Conv2d/BGEMM 的 optimized 分支
目前是真实存在的，不能一起删除。

### P2：转换 nn.Conv2d/nn.Linear 子类可能丢失自定义语义

当前用 isinstance 选择目标，因此自定义 Conv2d/Linear 子类也会被替换，只复制
标准 geometry、weight、bias 和训练状态。自定义 forward、forward hooks、
parametrization 以及额外属性不会迁移。建议默认只转换 type(module) is nn.Conv2d
或 nn.Linear，并提供 predicate/factory 给高级用户显式转换子类。

### P2：跨模块的 Parameter tying 不会保留

本轮已修复“同一个 module 被多个父属性引用”的 alias；但两个不同 module 若
共享同一个 weight Parameter，转换后仍会各自创建 Parameter，权重绑定会断开。
建议在有 tied-weight 模型需求时增加 Parameter identity 映射。

### P3：公开命名别名较多

每种浮点层同时暴露 float16/fp16、bfloat16/bf16 的长短名称，例如
Conv2d_float16 与 Conv2d_fp16 指向同一个类，functional API 也同样重复。
这些接口是明确的兼容别名，但从 API 面看确实冗余。建议选 fp16/bf16 为规范名称，
长名称保留一个 deprecation 周期后再删除，避免立即破坏现有调用方。

### P3：两个 Linear 文件存在可维护性重复

FP16/BF16 Linear 的校验、reshape、autograd 和 Module 框架基本相同，仅 dtype、
LUT side 和 backend op 不同。可以像 Conv2d 一样抽取私有 shared base/helper，
同时继续保留两个独立公开类和各自明确的 repr。不要重新引入对用户可见的通用
Float16OrBFloat16 类，否则会削弱这次要求的类型边界。

### P3：性能数据需要重新测量

uint32 是本轮明确要求，但它把 FP16 LUT 从 2 MiB 增至 4 MiB、BF16 从 32 KiB
增至 64 KiB。现有 delivery 文档中的历史 benchmark 来自旧存储条件，不能直接
代表 uint32 ABI 的性能。正确性已经重测；发布性能数字前应重新跑 benchmark，
尤其关注 FP16 LUT cache miss 与 memory bandwidth。

## 推荐后续顺序

1. 先实现或明确放弃 grouped/depthwise Conv2d。
2. 决定 Linear optimized 是实现真实优化还是移除接口。
3. 定义对子类、hooks、parametrization 与 tied Parameter 的转换策略。
4. 确定规范公开命名，并设计兼容 alias 的 deprecation。
5. 在 uint32 ABI 下重新跑并更新性能数据。
