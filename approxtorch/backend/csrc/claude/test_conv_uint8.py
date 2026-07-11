# Correctness tests for Conv2d_uint8 (uint8 x uint8 multiplier, static
# asymmetric quantization: per-tensor activation, per-channel weight; ste).
#
#  A. exact-product LUT (lut[a][b] = a*b): module forward must equal a
#     centered integer conv F.conv2d((q_x - z_x), (q_w - z_w)) BITWISE
#     (geometries keep every intermediate < 2^24, so fp32 sums are exact
#     in any order; conv zero-pad of the centered image == module z_x-pad)
#  B. random small-int LUT: module forward must equal an independent
#     unfold+gather oracle with the same zero-point corrections, BITWISE
#  C. ste grads vs torch autograd through an F.conv2d surrogate built on the
#     same fake-quant chain (exact-product LUT -> identical function of x/w)
#  D. eval/no-grad forward bit-identical to the training-mode values
#  E. EMA min/max buffer updates move with data; freeze_scale stops them
#  F. learning smoke: single layer overfits a fixed target (loss drops)
#
# geometry sweep: 1x1 fast path (flatten / stride slice / +padding), 3x3,
# stride 2, dilation 2, 5x5, rectangular kernel+padding with H != W,
# bias / no-bias, z_x != 0 (signed data) and z_x == 0 (ReLU data).
import torch
import torch.nn.functional as F
import approxtorch as at
from approxtorch.nn import quantization
from approxtorch.nn.Conv2d_uint8 import Conv2d_uint8, uint8_qparams

torch.manual_seed(0)
dev = 'cuda'
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
nfail = 0


def report(ok, msg):
    global nfail
    if not ok:
        nfail += 1
    print(f'{"PASS" if ok else "FAIL"} {msg}')


def relerr(a, b):
    b = b.double()
    return ((a.double() - b).norm() / (b.norm() + 1e-30)).item()


aa = torch.arange(256, device=dev).float()
prod_lut = (aa.view(-1, 1) * aa.view(1, -1)).reshape(-1)     # exact a*b
rand_lut = torch.randint(-64, 64, (65536,), device=dev).float()


def make_module(C, O, k, s, p, d, lut, bias, x_range=(-0.9, 1.1)):
    m = Conv2d_uint8(C, O, k, lut, bias=bias, stride=s, padding=p,
                     dilation=d).to(dev)
    m.train()
    m.freeze_scale()
    with torch.no_grad():
        m.x_min.fill_(x_range[0])
        m.x_max.fill_(x_range[1])
    return m


def qchain(m, x):
    s_x, z_x = uint8_qparams(m.x_min, m.x_max)
    s_w, z_w = uint8_qparams(m.w_min, m.w_max)
    xq = quantization.static_quantize_uint8(x, s_x, z_x)
    wq = quantization.static_quantize_uint8(m.weight, s_w, z_w, ch_axis=0)
    return xq, wq, s_x, z_x, s_w, z_w


def dequant_like_module(m, y_int, s_x, s_w):
    s = (s_x * s_w).view(1, -1, 1, 1)
    if m.bias is not None:
        return torch.addcmul(m.bias.view(1, -1, 1, 1), y_int, s)
    return y_int * s


# unfold+gather LUT oracle with zero-point corrections, all in double
def oracle_gather(m, x, lut):
    xq, wq, s_x, z_x, s_w, z_w = qchain(m, x)
    B, C, H, W = x.shape
    O = m.out_channels
    pH, pW = m.padding
    if pH or pW:
        xp = torch.full((B, C, H + 2 * pH, W + 2 * pW), z_x.item(), device=dev)
        xp[:, :, pH:pH + H, pW:pW + W] = xq
        xq = xp
    xu = F.unfold(xq, m.kernel_size, dilation=m.dilation, padding=0,
                  stride=m.stride).long()                    # (B, K, L)
    wu = wq.view(O, -1).long()                               # (O, K)
    lut2d = lut.view(256, 256).double()
    lutsum = lut2d[xu.unsqueeze(1), wu.unsqueeze(0).unsqueeze(-1)].sum(dim=2)
    K = wu.shape[1]
    y_int = (lutsum
             - z_w.double().view(1, -1, 1) * xu.sum(dim=1, keepdim=True).double()
             - (z_x.double() * wu.sum(dim=1).double()).view(1, -1, 1)
             + K * z_x.double() * z_w.double().view(1, -1, 1))
    sH, sW = m.stride
    dH, dW = m.dilation
    kH, kW = m.kernel_size
    OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return dequant_like_module(m, y_int.view(B, O, OH, OW).float(), s_x, s_w)


# geometry sweep: (C, O, k, s, p, d, bias) — K = C*kh*kw kept <= 75 so every
# intermediate (lut sums and correction terms) stays < 2^24 -> fp32-exact
geoms = [
    (8, 16, 1,      1, 0,      1, True),    # 1x1 fast path (flatten)
    (8, 16, 1,      2, 0,      1, True),    # 1x1 fast path (stride slice)
    (8, 16, 1,      1, 1,      1, True),    # 1x1 + padding
    (8, 16, 3,      1, 1,      1, True),    # classic 3x3
    (6, 12, 3,      2, 2,      1, False),   # stride 2 / pad 2 / no bias
    (6, 12, 3,      1, 1,      2, True),    # dilation 2
    (3, 10, 5,      2, 2,      1, True),    # 5x5
    (5, 9,  (3, 5), 1, (1, 2), 1, True),    # rect kernel/pad, H != W
]

for gi, (C, O, k, s, p, d, bias) in enumerate(geoms):
    H, W = (13, 17) if isinstance(k, tuple) else (14, 14)
    x = torch.randn(3, C, H, W, device=dev) * 0.5
    tag = f'C{C} O{O} k{k} s{s} p{p} d{d} bias={bias}'

    # ---- A: exact-product LUT vs centered integer conv (bitwise) ----
    m = make_module(C, O, k, s, p, d, prod_lut, bias)
    y = m(x)
    xq, wq, s_x, z_x, s_w, z_w = qchain(m, x)
    y_int = F.conv2d((xq - z_x).double(), (wq - z_w.view(-1, 1, 1, 1)).double(),
                     None, m.stride, m.padding, m.dilation)
    y_ref = dequant_like_module(m, y_int.float(), s_x, s_w)
    report(torch.equal(y, y_ref), f'A exact-lut fwd  {tag}')

    # ---- C: ste grads vs autograd surrogate (same fake-quant chain) ----
    g = torch.randn_like(y)
    x1 = x.clone().requires_grad_()
    m.zero_grad()
    (m(x1) * g).sum().backward()
    x2 = x.clone().requires_grad_()
    w2 = m.weight.detach().clone().requires_grad_()
    xq2 = quantization.static_quantize_uint8(x2, s_x, z_x)
    wq2 = quantization.static_quantize_uint8(w2, s_w, z_w, ch_axis=0)
    y2 = F.conv2d(xq2 - z_x, wq2 - z_w.view(-1, 1, 1, 1), None,
                  m.stride, m.padding, m.dilation) * (s_x * s_w).view(1, -1, 1, 1)
    if bias:
        b2 = m.bias.detach().clone().requires_grad_()
        y2 = y2 + b2.view(1, -1, 1, 1)
    (y2 * g).sum().backward()
    ex = relerr(x1.grad, x2.grad)
    ew = relerr(m.weight.grad, w2.grad)
    ok = ex < 1e-5 and ew < 1e-5
    if bias:
        eb = relerr(m.bias.grad, b2.grad)
        ok = ok and eb < 1e-5
    report(ok, f'C ste grads      {tag}  gx {ex:.1e} gw {ew:.1e}'
               + (f' gb {eb:.1e}' if bias else ''))

    # ---- B: random LUT vs unfold+gather oracle (bitwise) ----
    m2 = make_module(C, O, k, s, p, d, rand_lut, bias)
    y = m2(x)
    report(torch.equal(y, oracle_gather(m2, x, rand_lut)),
           f'B rand-lut fwd   {tag}')

    # ---- D: eval/no-grad forward bit-identical (first geometry only) ----
    if gi == 0:
        m.eval()
        with torch.no_grad():
            y_eval = m(x)
        report(torch.equal(y_eval, y_ref), 'D eval/no-grad forward bit-identical')

# z_x == 0 corner (ReLU-style data, real 0 quantizes to index 0)
C, O = 6, 12
x = torch.relu(torch.randn(3, C, 14, 14, device=dev))
m = make_module(C, O, 3, 1, 1, 1, prod_lut, True, x_range=(0.0, 1.2))
y = m(x)
xq, wq, s_x, z_x, s_w, z_w = qchain(m, x)
assert z_x.item() == 0.0
y_int = F.conv2d((xq - z_x).double(), (wq - z_w.view(-1, 1, 1, 1)).double(),
                 None, m.stride, m.padding, m.dilation)
report(torch.equal(y, dequant_like_module(m, y_int.float(), s_x, s_w)),
       'A exact-lut fwd  ReLU data (z_x = 0)')

# ------------------------------------------------------------------------ E
m = make_module(4, 8, 3, 1, 1, 1, rand_lut, True)
m.unfreeze_scale()
xm0, xM0 = m.x_min.item(), m.x_max.item()
wm0 = m.w_min.clone()
# w_min/w_max start exactly at the current weight stats (EMA fixed point),
# so move the weights first to see the w-side EMA track them
with torch.no_grad():
    m.weight.mul_(2.0)
_ = m(torch.randn(2, 4, 10, 10, device=dev) * 3.0)
moved = m.x_min.item() != xm0 and m.x_max.item() != xM0 \
        and not torch.equal(m.w_min, wm0)
report(moved, 'E EMA updates move min/max buffers')
m.freeze_scale()
xm1, xM1 = m.x_min.item(), m.x_max.item()
_ = m(torch.randn(2, 4, 10, 10, device=dev) * 5.0)
report(m.x_min.item() == xm1 and m.x_max.item() == xM1,
       'E freeze_scale stops updates')
m.eval()
m.unfreeze_scale()
xm2 = m.x_min.item()
with torch.no_grad():
    _ = m(torch.randn(2, 4, 10, 10, device=dev) * 5.0)
report(m.x_min.item() == xm2, 'E eval mode never updates')

# ------------------------------------------------------------------------ F
m = Conv2d_uint8(4, 8, 3, prod_lut, bias=True, padding=1).to(dev)
m.train()
x = torch.randn(4, 4, 12, 12, device=dev) * 0.5
target = torch.randn(4, 8, 12, 12, device=dev) * 0.1
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
l0 = None
for step in range(100):
    opt.zero_grad()
    loss = F.mse_loss(m(x), target)
    if l0 is None:
        l0 = loss.item()
    loss.backward()
    opt.step()
report(loss.item() < 0.3 * l0, f'F learning smoke: loss {l0:.4f} -> {loss.item():.4f}')

print(f'\n{"ALL PASS" if nfail == 0 else f"{nfail} FAILURES"}')
raise SystemExit(1 if nfail else 0)
