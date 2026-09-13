"""BF16 custom-gradient checks: python -m pytest -q <this file>."""

import pytest
import torch
import torch.nn.functional as F

from approxtorch import convert_float_model
from approxtorch.backend import ops
from approxtorch.nn import (
    Conv2d_bfloat16,
    Linear_bfloat16,
    approx_mul_bf16_custom,
    bgemm_bf16_custom,
    conv2d_bfloat16,
    gemm_bf16_custom,
    linear_bfloat16,
)


cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is required"
)


def gradient_luts(device="cpu", *, exact=True):
    fraction = torch.arange(128, dtype=torch.float32, device=device)
    row, col = fraction[:, None], fraction[None, :]
    if exact:
        dx = (1 + col / 128).expand(128, 128)
        dw = (1 + row / 128).expand(128, 128)
    else:
        dx = 0.1234567 + row * 0.003 + col * 0.007
        dw = 0.4567891 - row * 0.004 + col * 0.002
    # The CUDA contract is two independent FP32 arrays indexed by (Fx << 7)|Fw.
    return dx.contiguous().view(-1), dw.contiguous().view(-1)


def forward_lut(device="cpu"):
    mantissa = 1 + torch.arange(128, dtype=torch.float32) / 128
    product_bits = (mantissa[:, None] * mantissa[None, :]).bfloat16()
    return (product_bits.view(torch.int16).int() - 0x3F80).to(
        device=device, dtype=torch.uint32
    )


def bf16_bits(bits, device="cuda"):
    return torch.as_tensor(bits).to(torch.int16).view(torch.bfloat16).to(device)


def normal_values(shape, seed=0, device="cuda", exponent_low=124, exponent_high=130):
    generator = torch.Generator().manual_seed(seed)
    sign = torch.randint(0, 2, shape, generator=generator)
    exponent = torch.randint(exponent_low, exponent_high, shape, generator=generator)
    fraction = torch.randint(0, 128, shape, generator=generator)
    return bf16_bits((sign << 15) | (exponent << 7) | fraction, device)


def pair_partials(x, w, dx_lut, dw_lut):
    """Raw-field oracle; operands may broadcast, reductions are caller-owned."""
    xb = x.contiguous().view(torch.int16).int() & 0xFFFF
    wb = w.contiguous().view(torch.int16).int() & 0xFFFF
    index = ((xb & 127) << 7) | (wb & 127)
    # Scale the LUT directly so an exponent of 128 need not materialize inf.
    dx = torch.ldexp(dx_lut[index].double(), ((wb >> 7) & 255) - 127)
    dw = torch.ldexp(dw_lut[index].double(), ((xb >> 7) & 255) - 127)
    dx = torch.where((wb & 0x8000) != 0, -dx, dx).float()
    dw = torch.where((xb & 0x8000) != 0, -dw, dw).float()
    nonzero = ((xb & 0x7FFF) != 0) & ((wb & 0x7FFF) != 0)
    return torch.where(nonzero, dx, 0.0), torch.where(nonzero, dw, 0.0)


def assert_gradient(actual, expected):
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual, expected.bfloat16(), rtol=0.008, atol=1e-4, equal_nan=True
    )


@cuda
def test_exact_luts_match_pytorch_multiply_backward():
    x = normal_values((4096,), 1, exponent_low=100, exponent_high=150)
    w = normal_values(x.shape, 2, exponent_low=100, exponent_high=150)
    x.requires_grad_()
    w.requires_grad_()
    grad_output = normal_values(x.shape, 3)
    gx, gw = gradient_luts("cuda")
    lut = forward_lut("cuda")
    output = approx_mul_bf16_custom(x, w, lut, gx, gw)
    torch.testing.assert_close(output, ops.approx_mul_bf16(x, w, lut), rtol=0, atol=0)
    actual = torch.autograd.grad(output, (x, w), grad_output)
    expected = torch.autograd.grad(x * w, (x, w), grad_output)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@cuda
def test_asymmetric_luts_all_fraction_pairs_and_signed_exponents():
    fraction = torch.arange(16384, dtype=torch.int32)
    fx, fw = fraction >> 7, fraction & 127
    x = bf16_bits(((fx & 1) << 15) | ((117 + fx % 21) << 7) | fx)
    w = bf16_bits(((fw & 1) << 15) | ((117 + fw % 21) << 7) | fw)
    g = normal_values(x.shape, 4)
    gx, gw = gradient_luts("cuda", exact=False)
    actual = ops.approx_mul_bf16_backward(x, w, g, gx, gw)
    for got, partial in zip(actual, pair_partials(x, w, gx, gw)):
        assert_gradient(got, g.float() * partial)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "odd-storage-offset"])
@pytest.mark.parametrize("need_x,need_w", [(True, True), (True, False), (False, True)])
def test_large_elementwise_pairs_tail_and_unaligned_fallback(dtype, offset, need_x, need_w):
    # Exercise the packed launch threshold, its scalar final element, and the
    # scalar fallback for a contiguous view with an unaligned BF16 address.
    count = 2 ** 20 + 1
    x = normal_values((count + offset,), 41)[offset:]
    w = normal_values((count + offset,), 42)[offset:]
    g = (normal_values((count + offset,), 43).float() + 0.1234567).to(dtype)[offset:]
    assert x.is_contiguous() and x.storage_offset() == offset
    assert x.data_ptr() % 4 == 2 * offset
    # Give the odd final element a distinct sign, exponent, and fraction so
    # accidentally omitting or duplicating the scalar tail cannot pass.
    x[-1] = bf16_bits(0xBDAB)
    w[-1] = bf16_bits(0x40F1)
    g[-1] = -0.8125
    gx, gw = gradient_luts("cuda", exact=False)
    actual = ops.approx_mul_bf16_backward(x, w, g, gx, gw, need_x, need_w)
    for needed, got, partial in zip((need_x, need_w), actual, pair_partials(x, w, gx, gw)):
        if needed:
            # No reduction: the FP32 bit-field oracle must round identically.
            torch.testing.assert_close(got, (g.float() * partial).bfloat16(), rtol=0, atol=0)
        else:
            assert got is None


@cuda
def test_special_values_follow_raw_fields_and_zero_branch():
    # Forward has no IEEE NaN/Inf propagation or subnormal normalization/FTZ.
    x = bf16_bits([0, 0x8000, 1, 0x7F, 0x8001, 0x807F,
                   0x7F80, 0xFF80, 0x7FC1, 0xFFFF, 0x0080, 0x7F7F])
    w = bf16_bits([0xBF91] * x.numel())
    gx = torch.full((16384,), 0.25, device="cuda")
    gw = torch.full_like(gx, 0.375)
    g = torch.ones_like(x)
    actual = ops.approx_mul_bf16_backward(x, w, g, gx, gw)
    for got, want in zip(actual, pair_partials(x, w, gx, gw)):
        torch.testing.assert_close(got, want.bfloat16(), rtol=0, atol=0)

    zero_x = bf16_bits([0, 0x8000, 0x3F80, 0xBF80])
    zero_w = bf16_bits([0x3F80, 0xBF80, 0, 0x8000])
    g = bf16_bits([0x7F80, 0x7FC1, 0xFF80, 0x7FC1])
    actual = ops.approx_mul_bf16_backward(zero_x, zero_w, g, gx, gw)
    for got in actual:
        assert torch.count_nonzero(got.view(torch.int16)).item() == 0


@cuda
@pytest.mark.parametrize("kind", ["gemm", "bgemm"])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_matrix_backward_reductions_and_forward_parity(kind, optimized, exact):
    gx, gw = gradient_luts("cuda", exact=exact)
    lut = forward_lut("cuda")
    if kind == "gemm":
        x = normal_values((3, 5), 5).requires_grad_()
        w = normal_values((5, 35), 6).requires_grad_()
        g = normal_values((3, 35), 7)
        custom = gemm_bf16_custom
        forward = ops.gemm_bf16 if optimized else ops.gemm_bf16_naive
        dx, dw = pair_partials(x[:, :, None], w[None, :, :], gx, gw)
        expected = ((g.float()[:, None, :] * dx).sum(2),
                    (g.float()[:, None, :] * dw).sum(0))
    else:
        # N*L exceeds the 256-thread weight block, including a partial tail.
        x = normal_values((3, 5, 97), 5).requires_grad_()
        w = normal_values((7, 5), 6).requires_grad_()
        g = normal_values((3, 7, 97), 7)
        custom = bgemm_bf16_custom
        forward = ops.bgemm_bf16 if optimized else ops.bgemm_bf16_naive
        dx, dw = pair_partials(x[:, None, :, :], w[None, :, :, None], gx, gw)
        expected = ((g.float()[:, :, None, :] * dx).sum(1),
                    (g.float()[:, :, None, :] * dw).sum((0, 3)))
    output = custom(x, w, lut, gx, gw, optimized=optimized)
    torch.testing.assert_close(output, forward(x, w, lut), rtol=0, atol=0)
    for got, want in zip(torch.autograd.grad(output, (x, w), g), expected):
        assert_gradient(got, want)


def matrix_partials_reference(kind, x, w, g, gx, gw):
    if kind == "gemm":
        dx, dw = pair_partials(x[:, :, None], w[None, :, :], gx, gw)
        return ((g.float()[:, None, :] * dx).sum(2),
                (g.float()[:, None, :] * dw).sum(0))
    dx, dw = pair_partials(x[:, None, :, :], w[None, :, :, None], gx, gw)
    return ((g.float()[:, :, None, :] * dx).sum(1),
            (g.float()[:, :, None, :] * dw).sum((0, 3)))


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind,xs,ws,gs", [
    pytest.param("gemm", (16, 33), (33, 33), (16, 33), id="gemm-native-short"),
    pytest.param("gemm", (64, 33), (33, 33), (64, 33), id="gemm-native-original-table-64"),
    pytest.param("gemm", (32, 128), (128, 64), (32, 64), id="gemm-native-32"),
    pytest.param("gemm", (64, 64), (64, 128), (64, 128), id="gemm-native-64"),
    pytest.param("gemm", (33, 65), (65, 129), (33, 129), id="gemm-transposed-64"),
    pytest.param("gemm", (65, 65), (65, 65), (65, 65), id="gemm-transposed-128"),
    pytest.param("gemm", (129, 65), (65, 65), (129, 65), id="gemm-tiled-x-full-w"),
    pytest.param("gemm", (67, 33), (33, 33), (67, 33), id="gemm-transposed-single-output"),
    pytest.param("bgemm", (2, 33, 15), (33, 33), (2, 33, 15), id="bgemm-short-32"),
    pytest.param("bgemm", (2, 33, 17), (33, 33), (2, 33, 17), id="bgemm-short-64"),
    pytest.param("bgemm", (2, 33, 33), (33, 33), (2, 33, 33), id="bgemm-short-128"),
    pytest.param("bgemm", (3, 129, 45), (65, 129), (3, 65, 45), id="bgemm-full-parts"),
    pytest.param("bgemm", (3, 33, 45), (33, 33), (3, 33, 45), id="bgemm-original-table"),
    pytest.param("bgemm", (2, 129, 513), (65, 129), (2, 65, 513), id="bgemm-long-tail"),
])
def test_matrix_optimized_layouts_match_bit_field_reference(kind, xs, ws, gs, dtype):
    # Non-multiple dimensions cover the output-channel, K, spatial, and warp
    # tails across the small and large reduction paths. Each LUT depends on
    # both operands, so swapping rows/columns or the two tables is observable.
    x, w = normal_values(xs, 44), normal_values(ws, 45)
    g = (normal_values(gs, 46).float() + 0.1234567).to(dtype)
    gx, gw = gradient_luts("cuda", exact=False)
    actual = getattr(ops, f"{kind}_bf16_backward")(x, w, g, gx, gw)
    for got, want in zip(actual, matrix_partials_reference(kind, x, w, g, gx, gw)):
        assert_gradient(got, want)


@cuda
def test_matrix_transpose_scratch_follows_current_cuda_stream():
    x, w = normal_values((65, 65), 47), normal_values((65, 65), 48)
    final_gradient = normal_values((65, 65), 49).float() + 0.1234567
    g = torch.zeros_like(final_gradient)
    gx, gw = gradient_luts("cuda", exact=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        g.copy_(final_gradient)
        actual = ops.gemm_bf16_backward(x, w, g, gx, gw)
        # Scratch lifetimes end at the host return, before the asynchronous
        # kernels complete. Reusing that allocator storage must remain ordered.
        torch.empty_like(x).fill_(float("nan"))
        torch.empty_like(g).fill_(float("nan"))
        torch.empty_like(gw).fill_(float("nan"))
    stream.synchronize()
    expected = matrix_partials_reference("gemm", x, w, final_gradient, gx, gw)
    for got, want in zip(actual, expected):
        assert_gradient(got, want)


@cuda
@pytest.mark.parametrize("need_x,need_w", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("kind", ["approx_mul", "gemm", "bgemm"])
def test_only_requested_gradients(kind, need_x, need_w):
    shapes = {"approx_mul": ((7,), (7,), (7,)),
              "gemm": ((3, 5), (5, 7), (3, 7)),
              "bgemm": ((2, 5, 7), (3, 5), (2, 3, 7))}
    xs, ws, gs = shapes[kind]
    x, w, g = normal_values(xs, 8), normal_values(ws, 9), normal_values(gs, 10)
    gx, gw = gradient_luts("cuda", exact=False)
    backward = getattr(ops, f"{kind}_bf16_backward")
    expected = backward(x, w, g, gx, gw)
    actual = backward(x, w, g, gx, gw, need_x, need_w)
    for needed, got, want in zip((need_x, need_w), actual, expected):
        if needed:
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        else:
            assert got is None
    custom = {"approx_mul": approx_mul_bf16_custom,
              "gemm": gemm_bf16_custom, "bgemm": bgemm_bf16_custom}[kind]
    output = custom(x.requires_grad_(need_x), w.requires_grad_(need_w),
                    forward_lut("cuda"), gx, gw)
    assert output.requires_grad == (need_x or need_w)
    if output.requires_grad:
        output.backward(g)
        for needed, operand, want in zip((need_x, need_w), (x, w), expected):
            if needed:
                torch.testing.assert_close(operand.grad, want, rtol=0, atol=0)
            else:
                assert operand.grad is None


@cuda
@pytest.mark.parametrize("kind", ["approx_mul", "gemm", "bgemm"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["strided", "expanded"])
def test_upstream_dtype_and_noncontiguous_layout(kind, dtype, layout):
    shapes = {"approx_mul": ((3, 7), (3, 7), (3, 7)),
              "gemm": ((3, 5), (5, 7), (3, 7)),
              "bgemm": ((2, 5, 7), (3, 5), (2, 3, 7))}
    xs, ws, gs = shapes[kind]
    x, w = normal_values(xs, 16), normal_values(ws, 17)
    if layout == "expanded":
        g = torch.tensor(0.1234567, device="cuda", dtype=dtype).expand(gs)
    else:
        g = (normal_values((*gs[:-1], gs[-1] * 2), 18).float() + 0.1234567)
        g = g.to(dtype)[..., ::2]
    assert not g.is_contiguous()
    gx, gw = gradient_luts("cuda")
    actual = getattr(ops, f"{kind}_bf16_backward")(x, w, g, gx, gw)
    if kind == "approx_mul":
        expected = g.float() * w.float(), g.float() * x.float()
    elif kind == "gemm":
        expected = g.float() @ w.float().t(), x.float().t() @ g.float()
    else:
        expected = (torch.einsum("nol,ok->nkl", g.float(), w.float()),
                    torch.einsum("nol,nkl->ok", g.float(), x.float()))
    for got, want in zip(actual, expected):
        assert_gradient(got, want)


@cuda
@pytest.mark.parametrize("kind,xs,ws,gs", [
    ("approx_mul", (0,), (0,), (0,)),
    ("gemm", (0, 5), (5, 7), (0, 7)),
    ("gemm", (3, 0), (0, 7), (3, 7)),
    ("gemm", (3, 5), (5, 0), (3, 0)),
    ("gemm", (0, 65), (65, 65), (0, 65)),
    ("bgemm", (0, 5, 7), (3, 5), (0, 3, 7)),
    ("bgemm", (2, 0, 7), (3, 0), (2, 3, 7)),
    ("bgemm", (2, 5, 0), (3, 5), (2, 3, 0)),
    ("bgemm", (2, 5, 7), (0, 5), (2, 0, 7)),
    ("bgemm", (2, 65, 0), (65, 65), (2, 65, 0)),
])
def test_empty_dimensions(kind, xs, ws, gs):
    x, w, g = normal_values(xs, 19), normal_values(ws, 20), normal_values(gs, 21)
    gx, gw = gradient_luts("cuda")
    actual = getattr(ops, f"{kind}_bf16_backward")(x, w, g, gx, gw)
    for got, operand in zip(actual, (x, w)):
        torch.testing.assert_close(got, torch.zeros_like(operand), rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("kind", ["linear", "conv"])
@pytest.mark.parametrize("optimized", [False, True])
def test_layers_exact_gradient_and_unchanged_forward(kind, optimized):
    gx, gw = gradient_luts("cuda")
    lut = forward_lut("cuda")
    if kind == "linear":
        module = Linear_bfloat16(5, 7, lut, optimized=optimized,
                                grad_x_lut=gx, grad_w_lut=gw)
        x = normal_values((2, 3, 5), 11).requires_grad_()
        forward, reference = linear_bfloat16, F.linear
    else:
        # No overlap or padded zeros: the existing BF16 col2im rounding does
        # not introduce a second reduction roundoff in this comparison.
        module = Conv2d_bfloat16(2, 3, 2, lut, stride=2, optimized=optimized,
                                grad_x_lut=gx, grad_w_lut=gw)
        x = normal_values((2, 2, 6, 8), 11).requires_grad_()
        forward = lambda a, w, table, b, **kw: conv2d_bfloat16(
            a, w, table, b, stride=2, **kw)
        reference = lambda a, w, b: F.conv2d(a, w, b, stride=2)
    with torch.no_grad():
        module.weight.copy_(normal_values(module.weight.shape, 12))
        module.bias.copy_(normal_values(module.bias.shape, 13))
    output = module(x)
    old_output = forward(x, module.weight, lut, module.bias, optimized=optimized)
    torch.testing.assert_close(output, old_output, rtol=0, atol=0)
    g = normal_values(output.shape, 14)
    actual = torch.autograd.grad(output, (x, module.weight, module.bias), g)
    refs = [value.detach().float().requires_grad_()
            for value in (x, module.weight, module.bias)]
    expected = torch.autograd.grad(reference(*refs), refs, g.float())
    for got, want in zip(actual, expected):
        assert_gradient(got, want)


@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_modules_keep_fixed_fp32_buffers_on_dtype_conversion(kind):
    gx, gw = gradient_luts(exact=False)
    cls, args = ((Linear_bfloat16, (3, 5)) if kind == "linear"
                 else (Conv2d_bfloat16, (2, 3, 2)))
    module = cls(*args, forward_lut(), grad_x_lut=gx, grad_w_lut=gw)
    for convert in (lambda m: m.bfloat16(), lambda m: m.to(dtype=torch.float16)):
        convert(module)
        for name, expected in (("grad_x_lut", gx), ("grad_w_lut", gw)):
            buffer = dict(module.named_buffers())[name]
            assert buffer.shape == (16384,)
            assert buffer.dtype == torch.float32 and not buffer.requires_grad
            assert name not in dict(module.named_parameters())
            torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
            torch.testing.assert_close(module.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["missing_pair", "matrix", "size", "dtype",
                                  "strided", "trainable"])
def test_gradient_lut_validation(bad):
    gx, gw = gradient_luts()
    if bad == "missing_pair":
        gw = None
    elif bad == "matrix":
        gx = gx.view(128, 128)
    elif bad == "size":
        gx = gx[:-1]
    elif bad == "dtype":
        gx = gx.bfloat16()
    elif bad == "strided":
        gx = gx.repeat_interleave(2)[::2]
    elif bad == "trainable":
        gx.requires_grad_()
    with pytest.raises((ValueError, TypeError), match="grad_"):
        Linear_bfloat16(3, 5, forward_lut(), grad_x_lut=gx, grad_w_lut=gw)


def test_converter_shares_fixed_gradient_buffers():
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 2), torch.nn.Linear(3, 5))
    gx, gw = gradient_luts(exact=False)
    converted = convert_float_model(
        model, forward_lut(), qtype="bf16", ignore_first_conv=False,
        grad_x_lut=gx, grad_w_lut=gw,
    ).bfloat16()
    assert isinstance(converted[0], Conv2d_bfloat16)
    assert isinstance(converted[1], Linear_bfloat16)
    for layer in converted:
        for name, expected in (("grad_x_lut", gx), ("grad_w_lut", gw)):
            actual = getattr(layer, name)
            assert actual.data_ptr() == expected.data_ptr()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cuda
def test_raw_backward_rejects_cpu_gradient_lut():
    x = normal_values((3,), 15)
    gx, gw = gradient_luts()
    with pytest.raises(RuntimeError, match="CUDA|cuda"):
        ops.approx_mul_bf16_backward(x, x, x, gx, gw.cuda())
