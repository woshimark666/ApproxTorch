"""Correctness checks for the pair-wise custom gradient Python path.

The custom tables use signed-int8 coordinates:
    dx_lut[x + 128, w + 128] = d ApproxMul(x, w) / dx
    dw_lut[x + 128, w + 128] = d ApproxMul(x, w) / dw
"""

import torch
import torch.nn.functional as F

import approxtorch as at
from approxtorch.nn import bgemm_int8, quantization
from approxtorch.nn.Conv2d_int8 import Conv2d_int8


if not torch.cuda.is_available():
    print("SKIP: CUDA is not available")
    raise SystemExit(0)


torch.manual_seed(0)
device = "cuda"
failures = 0


def report(ok, message):
    global failures
    if not ok:
        failures += 1
    print(f'{"PASS" if ok else "FAIL"} {message}')


def relerr(actual, expected):
    expected = expected.double()
    return ((actual.double() - expected).norm()
            / (expected.norm() + 1e-30)).item()


# Deliberately non-separable and non-symmetric tables catch any x/w swap or
# accidental reduction to a one-dimensional LRE-style table.
row = torch.arange(256, device=device, dtype=torch.float32).view(256, 1)
col = torch.arange(256, device=device, dtype=torch.float32).view(1, 256)
dx_pair = row * 1009.0 + col * 7.0 + (row * col).remainder(31.0)
dw_pair = row * -11.0 + col * 1013.0 + (row + 3.0 * col).remainder(37.0)


print("--- optimized CUDA pair indexing vs naive CUDA oracle ---")
for n, k, o, length in [
        (1, 1, 1, 1),
        (2, 7, 5, 9),
        (3, 33, 17, 65),
        (2, 65, 9, 31),
]:
    x = torch.randint(0, 256, (n, k, length), device=device,
                      dtype=torch.uint8)
    w = torch.randint(0, 256, (o, k), device=device, dtype=torch.uint8)
    go = torch.randn(n, o, length, device=device)
    gx_ref, gw_ref = at.backend.ops.bgemm_custom_grad_uint8_naive(
        x, w, go, dx_pair, dw_pair)
    gx = at.backend.ops.bgemm_custom_grad_uint8_dx(x, w, go, dx_pair)
    gw = at.backend.ops.bgemm_custom_grad_uint8_dw(x, w, go, dw_pair)
    ex, ew = relerr(gx, gx_ref), relerr(gw, gw_ref)
    report(ex < 2e-5 and ew < 2e-5,
           f"N={n} K={k} O={o} L={length}: gx={ex:.2e}, gw={ew:.2e}")


print("--- BGEMM autograd with exact-product custom derivatives ---")
signed = torch.arange(-128, 128, device=device, dtype=torch.float32)
exact_lut = signed.view(-1, 1) * signed.view(1, -1)
dx_exact = signed.view(1, -1).expand(256, 256).contiguous()
dw_exact = signed.view(-1, 1).expand(256, 256).contiguous()

x = torch.randint(-127, 128, (2, 17, 13), device=device).float()
w = torch.randint(-127, 128, (11, 17), device=device).float()
x.requires_grad_(True)
w.requires_grad_(True)
go = torch.randn(2, 11, 13, device=device)
y = bgemm_int8.bgemm_int8_custom(x, w, exact_lut, dx_exact, dw_exact)
y.backward(go)
gx_ref = torch.einsum("nol,ok->nkl", go, w.detach())
gw_ref = torch.einsum("nol,nkl->ok", go, x.detach())
ex, ew = relerr(x.grad, gx_ref), relerr(w.grad, gw_ref)
report(ex < 2e-5 and ew < 2e-5,
       f"BGEMM exact derivative: gx={ex:.2e}, gw={ew:.2e}")


print("--- Conv2d custom path vs exact PyTorch surrogate ---")
for kernel, stride, padding, dilation in [
        (1, 1, 0, 1),
        (3, 1, 1, 1),
        (3, 2, 2, 2),
]:
    module = Conv2d_int8(
        3, 5, kernel, exact_lut, grad="custom",
        dx=dx_exact, dw=dw_exact, bias=True,
        stride=stride, padding=padding, dilation=dilation,
    ).to(device).eval()
    with torch.no_grad():
        module.scale_x.fill_(0.025)

    x_value = torch.randn(2, 3, 13, 15, device=device)
    x_custom = x_value.detach().requires_grad_(True)
    module.weight.grad = None
    module.bias.grad = None
    y_custom = module(x_custom)
    go = torch.randn_like(y_custom)
    y_custom.backward(go)
    gx_custom = x_custom.grad.detach().clone()
    gw_custom = module.weight.grad.detach().clone()
    gb_custom = module.bias.grad.detach().clone()

    x_ref = x_value.detach().requires_grad_(True)
    w_ref = module.weight.detach().clone().requires_grad_(True)
    b_ref = module.bias.detach().clone().requires_grad_(True)
    xq = quantization.static_quantize_int8(
        x_ref, module.scale_x, module.qmin, module.qmax)
    qmax_w = 2 ** (module.weight_bits - 1) - 1
    wq = quantization.static_quantize_int8(
        w_ref, module.scale_w, -qmax_w, qmax_w, ch_axis=0)
    y_ref = F.conv2d(
        xq, wq, None, stride, padding, dilation, 1)
    y_ref = torch.addcmul(
        b_ref.view(1, -1, 1, 1), y_ref,
        (module.scale_x * module.scale_w).view(1, -1, 1, 1))
    y_ref.backward(go)

    ex = relerr(gx_custom, x_ref.grad)
    ew = relerr(gw_custom, w_ref.grad)
    eb = relerr(gb_custom, b_ref.grad)
    report(ex < 2e-5 and ew < 2e-5 and eb < 2e-6,
           f"k={kernel} s={stride} p={padding} d={dilation}: "
           f"gx={ex:.2e}, gw={ew:.2e}, gb={eb:.2e}")


print("\nALL PASS" if failures == 0 else f"\n{failures} FAILURES")
raise SystemExit(0 if failures == 0 else 1)
