"""GPU regression tests for INT32 forward sums and custom backward devices."""

import pytest
import torch

from approxtorch.backend import ops


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=range(torch.cuda.device_count()))
def device(request):
    return torch.device("cuda", request.param)


def case_data(kind, device):
    """Adversarial sums: small answers after large prefixes, and final rounding."""
    if kind == "int16":
        entries = [32767, 1, -32767, 0]
        indices = [0] * 1024 + [1] + [2] * 1024
    elif kind == "uint16":
        entries = [65535, 0, 0, 0]
        indices = [0] * 1025
    elif kind == "int32":
        entries = [2**24, 1, -(2**24), 0]
        # A split boundary after 32 terms must preserve the first partial's +1.
        indices = [0, 1] + [3] * 30 + [2] + [3] * 32
    elif kind == "wrap":
        entries = [2**30, 0, 0, 0]
        indices = [0] * 65
    else:
        raise AssertionError(kind)
    lut = torch.zeros(256, 256, device=device)
    lut[:4] = torch.tensor(entries, device=device)[:, None]
    ix = torch.tensor(indices, device=device)
    # Independent int64 oracle, wrapped once, rounded to FP32 once.
    expected = lut[ix, 0].to(torch.int64).sum().to(torch.int32).float()
    return lut, ix.float(), expected


@pytest.mark.parametrize("kind", ["int16", "uint16", "int32", "wrap"])
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("layout", ["xmk", "nflat", "sflat"])
@pytest.mark.parametrize("split", [False, True])
def test_bgemm_accumulation(device, kind, unsigned, layout, split):
    lut, ix, expected = case_data(kind, device)
    # cfg 10 = 8x32 tiles. 128x384 yields exactly 192 blocks (no split);
    # 2x3 yields one natural block (split-K). K tails are deliberately odd.
    rows, cols = (2, 3) if split else (128, 384)
    if layout == "xmk":
        n, length, outputs, cfg = rows, 1, cols, 10
    elif layout == "nflat":
        n, length, outputs, cfg = 1, rows, cols, 10
    else:
        n, length, outputs, cfg = 1, cols, rows, 110
    offset = 0 if unsigned else 128
    x = (ix - offset)[None, :, None].expand(n, -1, length).contiguous()
    w = torch.full((outputs, ix.numel()), -float(offset), device=device)
    name = f"bgemm_fake_{'uint8' if unsigned else 'int8'}_forward_cuda_claude_cfg"
    actual = getattr(torch.ops.approxtorch, name)(x, w, lut, cfg)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected.expand_as(actual))


@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("cfg", [10, 110])
def test_gemm_split_partial_keeps_integer_precision(device, unsigned, cfg):
    lut, ix, expected = case_data("int32", device)
    offset = 0 if unsigned else 128
    a = (ix - offset)[None, :].expand(2, -1)
    b = torch.full((3, ix.numel()), -float(offset), device=device).t()
    name = f"gemm_fake_{'uint8' if unsigned else 'int8'}_forward_cuda_claude_cfg"
    actual = getattr(torch.ops.approxtorch, name)(a, b, lut, cfg)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected.expand_as(actual))


@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.int8, torch.uint8])
def test_depthwise_accumulation(device, kernel, input_dtype):
    lut = torch.zeros(256, 256, device=device)
    lut[0] = 2**24
    lut[1] = 1
    lut[2] = -(2**24)
    indices = torch.full((1, 2, kernel, kernel), 3, device=device)
    indices.flatten(2)[:, :, :3] = torch.tensor([0, 1, 2], device=device)
    x = (indices if input_dtype == torch.uint8 else indices - 128).to(input_dtype)
    w = torch.full((2, kernel * kernel), -128.0, device=device)
    actual, wq = ops.dwconv_fake_int8_claude(x, w, lut, kernel)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, torch.ones_like(actual))
    assert torch.equal(wq, torch.zeros_like(wq))


def test_reference_forward_uses_int32(device):
    lut, ix, expected = case_data("int32", device)
    actual = ops.bgemm_fake_int8_gpt(
        (ix - 128)[None, :, None], torch.full((1, ix.numel()), -128.0, device=device),
        lut.flatten())
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected.expand_as(actual))


FORWARDS = [
    "bgemm_fake_int8_claude", "bgemm_fake_uint8_claude",
    "bgemm_fake_int8_claude_save", "bgemm_fake_uint8_claude_save",
    "gemm_fake_int8_claude", "gemm_fake_uint8_claude",
    "gemm_fake_int8_claude_save", "gemm_fake_uint8_claude_save",
    "dwconv_fake_int8_claude", "bgemm_fake_int8_gpt",
]


def call_forward(name, lut):
    z = torch.zeros(1, device=lut.device)
    if name.startswith("dwconv"):
        return getattr(ops, name)(z.reshape(1, 1, 1, 1), z.reshape(1, 1), lut, 1)[0]
    x = z.reshape(1, 1, 1) if name.startswith("bgemm") else z.reshape(1, 1)
    out = getattr(ops, name)(x, z.reshape(1, 1), lut)
    return out[0] if name.endswith("_save") else out


@pytest.mark.parametrize("name", FORWARDS)
@pytest.mark.parametrize("value", [0.5, float("nan"), float("inf"), -float("inf"),
                                  2**31, -(2**31) - 256])
def test_invalid_forward_lut_is_recoverable(device, name, value):
    lut = torch.zeros(65536, device=device)
    lut[-1] = value  # Invalid even if this entry is not looked up.
    with pytest.raises(RuntimeError, match="finite integer values in the INT32 range"):
        call_forward(name, lut)
    # No device assert / poisoned CUDA context after rejection.
    lut[-1] = 0
    assert torch.count_nonzero(call_forward(name, lut)) == 0


@pytest.mark.parametrize("name", FORWARDS)
@pytest.mark.parametrize("value", [-(2**31), 2**31 - 128])
def test_int32_lut_boundaries(device, name, value):
    lut = torch.full((65536,), float(value), device=device)
    actual = call_forward(name, lut)
    assert torch.equal(actual, torch.full_like(actual, float(value)))


def backward_inputs(device, dtype):
    x = torch.randint(0, 256, (2, 9, 7), device=device).transpose(1, 2)
    w = torch.randint(0, 256, (7, 5), device=device).t()
    xi, wi = x.long(), w.long()
    if dtype == "int8":
        x, w = x - 128, w - 128
    x, w = x.to(getattr(torch, dtype)), w.to(getattr(torch, dtype))
    dy = torch.randn(2, 9, 5, device=device).transpose(1, 2)
    # Fractional gradient LUTs remain valid and must keep FP32 arithmetic.
    r = torch.arange(256, device=device, dtype=torch.float32)[:, None]
    c = r.t()
    lut = (r * 0.01 + c * 0.001 + (r * c).remainder(17) * 0.003).t()
    values = lut.double()[xi[:, None, :, :], wi[None, :, :, None]]
    terms = dy.double()[:, :, None, :] * values
    return [x, w, dy, lut], terms.sum(1).float(), terms.sum((0, 3)).float()


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("grad", ["dx", "dw"])
def test_backward_device_guard_and_stream(device, dtype, grad):
    original = torch.cuda.current_device()
    caller = (device.index + 1) % torch.cuda.device_count()
    stream = torch.cuda.Stream(device=device)
    try:
        with torch.cuda.stream(stream):
            args, dx, dw = backward_inputs(device, dtype)
            # Queue an input update behind a delay: a launch on another
            # device's/default stream must not race ahead of this write.
            torch.cuda._sleep(10_000_000)
            args[2].mul_(2)
            dx, dw = dx * 2, dw * 2
            torch.cuda.set_device(caller)
            actual = getattr(ops, f"bgemm_custom_grad_{dtype}_{grad}")(*args)
            assert torch.cuda.current_device() == caller
            assert actual.device == device
        stream.synchronize()
        torch.testing.assert_close(actual, dx if grad == "dx" else dw,
                                   rtol=2e-5, atol=2e-5)
    finally:
        torch.cuda.set_device(original)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("grad", ["dx", "dw"])
@pytest.mark.parametrize("argument", [1, 2, 3])
@pytest.mark.parametrize("destination", ["cpu", "other_gpu"])
def test_backward_rejects_mixed_devices(device, dtype, grad, argument, destination):
    if destination == "other_gpu" and torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    args, _, _ = backward_inputs(device, dtype)
    target = "cpu" if destination == "cpu" else f"cuda:{(device.index + 1) % torch.cuda.device_count()}"
    args[argument] = args[argument].to(target)
    message = "all be CUDA" if destination == "cpu" else "same CUDA device"
    with pytest.raises(RuntimeError, match=message):
        getattr(ops, f"bgemm_custom_grad_{dtype}_{grad}")(*args)
