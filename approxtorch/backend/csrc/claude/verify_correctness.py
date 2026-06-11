# Independent-oracle correctness verification for the conv (groups=1 +
# depthwise) forward and ste/lre backward paths. Unlike test_conv_decoupled
# (whose fp64 truth shares this author's reading of the semantics), every
# check here compares against an implementation we did NOT write:
#
#  A. exact-product LUT (lut[i][j] = (i-128)(j-128)) => the whole module
#     forward must equal a plain unfold+GEMM conv on the quantized tensors
#     EXACTLY (integer sums < 2^24 are exact in fp32 under any reduction
#     order; F.conv2d itself is NOT a valid bitwise oracle, see
#     conv_ref_exact below)
#  B. ste grads vs torch autograd through a F.conv2d surrogate graph
#  C. lre grads vs the ORIGINAL reference op bgemm_lre_backward
#     (cuda/bgemm_lre_backward.cu, pre-claude), per-channel for depthwise
#  D. adversarial padding: dw[128] = 1e4 (any padding-semantics slip blows
#     far past the tolerance)
#  E. randomized fuzz over geometry x {groups=1, depthwise} x {ste, lre}
#  F. int16-LUT edge values, fp32-LUT fallback, non-contiguous inputs
#  G. end-to-end learning: swapped MobileNetV2 overfits one batch
import torch
import torch.nn as nn
import torch.nn.functional as F
import approxtorch as at
from approxtorch.nn import fakequant, bgemm
from approxtorch.nn.Conv2d_int8_decoupled import Conv2d_int8

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


ii = torch.arange(256, device=dev).float() - 128
exact_lut = ii.view(-1, 1) * ii.view(1, -1)           # exact signed product
id_table = ii.clone()                                 # d(xw)/dx = w etc.


def make_module(C, O, k, s, p, d, g, mode, lut, dx=None, dw=None, bias=True):
    kw = {}
    if mode == 'lre':
        kw = dict(dx=dx if dx is not None else torch.randn(256, device=dev),
                  dw=dw if dw is not None else torch.randn(256, device=dev))
    m = Conv2d_int8(C, O, k, lut, grad=mode, bias=bias, stride=s, padding=p,
                    dilation=d, groups=g, **kw).to(dev)
    m.train()
    m.update_scale = False
    with torch.no_grad():
        m.scale_x.fill_(0.02)
    return m


# quantized tensors exactly as the module computes them (python chain is
# bit-identical to the fused kernels -- established in earlier rounds)
def chain(m, x):
    xq = fakequant.symmetric_static_quantize_int8_per_tensor(
        x, m.scale_x, None, m.qmin, m.qmax)
    wq, s_w = fakequant.symmetric_dynamic_quantize_int8_per_channel(
        m.weight, ch_axis=0, bits=m.weight_bits)
    return xq, wq, s_w


def dequant(m, y_raw, s_w):
    s = (m.scale_x * s_w).view(1, -1, 1, 1)
    if m.bias is not None:
        return torch.addcmul(m.bias.view(1, -1, 1, 1), y_raw, s)
    return y_raw * s


# plain unfold + GEMM convolution: ordered fp32 sums of products, which are
# EXACT for integer operands below 2^24 in any summation order. F.conv2d is
# deliberately NOT used as the bitwise oracle: cuDNN may pick Winograd/FFT
# (e.g. k5 s2), which are transform-based, not plain sums, and produce
# ~1e-7 rel deviations ON INTEGER INPUTS (observed 0.0625 abs on k5 s2).
def conv_ref_exact(xq, wq, s, p, d, g):
    B = xq.shape[0]
    O, _, kh, kw = wq.shape
    H, W = xq.shape[2], xq.shape[3]
    xu = F.unfold(xq, (kh, kw), dilation=d, padding=p, stride=s)
    L = xu.shape[-1]
    if g == 1:
        y = wq.view(O, -1) @ xu                      # [B, O, L]
    else:
        xu = xu.view(B, g, -1, L)
        wv = wq.view(g, O // g, -1)
        y = torch.einsum('gok,bgkl->bgol', wv, xu).reshape(B, O, L)
    OH = (H + 2*p - d*(kh-1) - 1)//s + 1
    OW = (W + 2*p - d*(kw-1) - 1)//s + 1
    return y.view(B, O, OH, OW)


# ---- A. exact-product LUT: forward must equal exact GEMM conv bitwise ---
print('--- A. exact-LUT forward == unfold+GEMM conv (bitwise) ---')
for (tag, C, O, k, s, p, d, g) in [
        ('g1 k3 s1 p1',   32, 48, 3, 1, 1, 1, 1),
        ('g1 k3 s2 p1',   48, 64, 3, 2, 1, 1, 1),
        ('g1 1x1',        64, 32, 1, 1, 0, 1, 1),
        ('g1 k5 p2',      16, 24, 5, 1, 2, 1, 1),
        ('g1 k3 d2 p2',   16, 24, 3, 1, 2, 2, 1),
        ('dw k3 s1 p1',   32, 32, 3, 1, 1, 1, 32),
        ('dw k3 s2 p1',   96, 96, 3, 2, 1, 1, 96),
        ('dw k5 p2',      16, 16, 5, 1, 2, 1, 16),
        ('dw k3 d2 p2',   24, 24, 3, 1, 2, 2, 24)]:
    for mode in ('ste', 'lre'):
        m = make_module(C, O, k, s, p, d, g, mode, exact_lut)
        x = torch.randn(4, C, 17, 19, device=dev)
        with torch.no_grad():
            y = m(x)
            xq, wq, s_w = chain(m, x)
            y_ref = dequant(m, conv_ref_exact(xq, wq, s, p, d, g), s_w)
        report(torch.equal(y, y_ref), f'A {tag} {mode}')

# ---- B. ste grads vs torch autograd surrogate ---------------------------
print('--- B. ste grads vs torch-autograd F.conv2d surrogate ---')
for (tag, C, O, k, s, p, d, g) in [
        ('g1 k3 s1 p1', 16, 24, 3, 1, 1, 1, 1),
        ('g1 1x1',      32, 16, 1, 1, 0, 1, 1),
        ('g1 k3 s2',    16, 24, 3, 2, 1, 1, 1),
        ('dw k3 s1 p1', 32, 32, 3, 1, 1, 1, 32),
        ('dw k3 s2 p1', 48, 48, 3, 2, 1, 1, 48),
        ('dw k5 p2',    16, 16, 5, 1, 2, 1, 16)]:
    lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
    m = make_module(C, O, k, s, p, d, g, 'ste', lut)
    x0 = torch.randn(3, C, 15, 13, device=dev)
    with torch.no_grad():
        go = torch.randn_like(m(x0))

    x = x0.detach().requires_grad_(True)
    m.weight.grad = None; m.bias.grad = None
    m(x).backward(go)
    gx_m, gw_m, gb_m = x.grad.clone(), m.weight.grad.clone(), m.bias.grad.clone()

    # surrogate: same fakequant Functions, torch's own conv2d + autograd.
    # ste's defining property: grads as if the multiply were exact.
    x = x0.detach().requires_grad_(True)
    m.weight.grad = None; m.bias.grad = None
    xq, wq, s_w = chain(m, x)
    y_sur = dequant(m, F.conv2d(xq, wq, None, (s, s), (p, p), (d, d), g), s_w)
    y_sur.backward(go)
    e = (relerr(gx_m, x.grad), relerr(gw_m, m.weight.grad), relerr(gb_m, m.bias.grad))
    report(max(e) < 1e-5, f'B {tag}: gx {e[0]:.1e} gw {e[1]:.1e} gb {e[2]:.1e}')

# ---- C. lre grads vs ORIGINAL reference op ------------------------------
print('--- C. lre grads vs original bgemm_lre_backward reference op ---')
for (tag, C, O, k, s, p, d, g) in [
        ('g1 k3 s1 p1', 16, 24, 3, 1, 1, 1, 1),
        ('g1 k3 s2 p1', 16, 24, 3, 2, 1, 1, 1),
        ('g1 k5 d2 p3', 8, 12, 5, 1, 3, 2, 1),
        ('dw k3 s1 p1', 16, 16, 3, 1, 1, 1, 16),
        ('dw k3 s2 p1', 24, 24, 3, 2, 1, 1, 24),
        ('dw k5 p2',    8, 8, 5, 1, 2, 1, 8)]:
    lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
    dx = torch.randn(256, device=dev)
    dwl = torch.randn(256, device=dev)
    B, H, W = 3, 14, 11
    kk = k * k
    OH = (H + 2*p - d*(k-1) - 1)//s + 1
    OW = (W + 2*p - d*(k-1) - 1)//s + 1
    x_img = torch.randint(-127, 128, (B, C, H, W), device=dev).float().requires_grad_(True)
    w2 = torch.randint(-127, 128, (O, (C//g)*kk), device=dev).float().requires_grad_(True)
    geom = ((k, k), (s, s), (p, p), (d, d), g)
    y = bgemm.conv2d_int8_lre(x_img, w2, lut, dx, dwl, geom)
    go = torch.randn(B, O, OH*OW, device=dev)
    y.backward(go.view_as(y))

    with torch.no_grad():
        xu = F.unfold(x_img.detach(), (k, k), dilation=d, padding=p, stride=s)
        if g == 1:
            gxu, gw = at.backend.ops.bgemm_lre_backward(
                go, xu, w2.detach().t().contiguous(), dx, dwl)
            gw_ref = gw.t()
        else:  # depthwise: reference op per channel
            xuv = xu.view(B, C, kk, -1)
            gxu = torch.empty_like(xuv)
            gw_ref = torch.empty(C, kk, device=dev)
            for c in range(C):
                gxc, gwc = at.backend.ops.bgemm_lre_backward(
                    go[:, c:c+1, :].contiguous(), xuv[:, c].contiguous(),
                    w2.detach()[c].view(kk, 1).contiguous(), dx, dwl)
                gxu[:, c] = gxc
                gw_ref[c] = gwc.view(kk)
            gxu = gxu.reshape(B, C*kk, -1)
        gx_ref = F.fold(gxu, (H, W), (k, k), dilation=d, padding=p, stride=s)
    e = (relerr(x_img.grad, gx_ref), relerr(w2.grad, gw_ref))
    report(max(e) < 1e-5, f'C {tag}: gx {e[0]:.1e} gw {e[1]:.1e}')

# ---- D. adversarial padding: dw[128] huge -------------------------------
print('--- D. adversarial padding (dw[128] = 1e4) ---')
for (tag, C, O, g) in [('g1', 8, 12, 1), ('dw', 12, 12, 12)]:
    lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
    dx = torch.randn(256, device=dev)
    dwl = torch.randn(256, device=dev)
    dwl[128] = 1e4
    k, s, p, d = 3, 1, 2, 1            # H=5, p=2: padding-dominated
    B, H, W = 2, 5, 5
    OH = OW = H + 2*p - k + 1
    x_img = torch.randint(-127, 128, (B, C, H, W), device=dev).float().requires_grad_(True)
    w2 = torch.randint(-127, 128, (O, (C//g)*9), device=dev).float().requires_grad_(True)
    y = bgemm.conv2d_int8_lre(x_img, w2, lut, dx, dwl, ((3,3),(1,1),(2,2),(1,1),g))
    go = torch.randn(B, O, OH*OW, device=dev)
    y.backward(go.view_as(y))
    # fp64 truth straight from the definition
    with torch.no_grad():
        xu = F.unfold(x_img.detach(), (3, 3), padding=2).double()
        xp = dwl.double()[(xu + 128).long()]
        if g == 1:
            gw_t = torch.einsum('nol,nkl->ok', go.double(), xp)
        else:
            gw_t = torch.einsum('ncl,nckl->ck', go.double(),
                                xp.view(B, C, 9, -1))
    report(relerr(w2.grad, gw_t) < 2e-6, f'D {tag}: gw {relerr(w2.grad, gw_t):.1e}')

# ---- E. randomized fuzz --------------------------------------------------
print('--- E. fuzz (30 random geometries, fwd bitwise vs exact-lut GEMM conv'
      ' + grads vs autograd/reference) ---')
import random
random.seed(7)
fuzz_fail = 0
for t in range(30):
    g1 = random.random() < 0.5
    k = random.choice([1, 2, 3, 4, 5])
    s = random.choice([1, 2, 3])
    p = random.choice([0, 1, 2])
    d = random.choice([1, 2]) if k > 1 else 1
    C = random.randint(1, 48)
    O = random.randint(1, 48) if g1 else C
    g = 1 if g1 else C
    mode = random.choice(['ste', 'lre'])
    bias = random.random() < 0.7
    keff = d * (k - 1) + 1
    H = random.randint(keff, 23)
    W = random.randint(keff, 23)
    if (H + 2*p - keff) < 0 or (W + 2*p - keff) < 0:
        continue
    B = random.randint(1, 4)
    m = make_module(C, O, k, s, p, d, g, mode, exact_lut, bias=bias)
    x = torch.randn(B, C, H, W, device=dev)
    with torch.no_grad():
        y = m(x)
        xq, wq, s_w = chain(m, x)
        y_ref = dequant(m, conv_ref_exact(xq, wq, s, p, d, g), s_w)
    ok = torch.equal(y, y_ref)
    if mode == 'ste':   # grads vs autograd surrogate (exact-lut forward)
        go = torch.randn_like(y)
        xx = x.detach().requires_grad_(True)
        m.weight.grad = None
        m(xx).backward(go)
        gxm, gwm = xx.grad.clone(), m.weight.grad.clone()
        xx = x.detach().requires_grad_(True)
        m.weight.grad = None
        xq, wq, s_w = chain(m, xx)
        dequant(m, F.conv2d(xq, wq, None, (s, s), (p, p), (d, d), g), s_w).backward(go)
        ok = ok and relerr(gxm, xx.grad) < 1e-5 and relerr(gwm, m.weight.grad) < 1e-5
    if not ok:
        fuzz_fail += 1
        print(f'  FUZZ FAIL: B{B} C{C} O{O} {H}x{W} k{k} s{s} p{p} d{d} g{g} {mode}')
report(fuzz_fail == 0, f'E fuzz: {30 - fuzz_fail}/30 configs clean')

# ---- F. LUT edges + non-contiguous inputs -------------------------------
print('--- F. edge cases ---')
x8 = torch.randint(-127, 128, (2, 8, 12, 12), device=dev, dtype=torch.int8)
w = torch.randint(-127, 128, (8, 9), device=dev).float()
lut_edge = torch.full((256, 256), 32767.0, device=dev)
lut_edge[::2] = -32767.0                      # int16 boundary values
y1, _ = at.backend.ops.dwconv_fake_int8_claude(x8, w, lut_edge, 3, 1, 1, 1)
lut_bad = lut_edge.clone(); lut_bad[0, 0] = 32768.0   # forces fp32 path
y2, _ = at.backend.ops.dwconv_fake_int8_claude(x8, w, lut_bad, 3, 1, 1, 1)
lutf = lut_edge.reshape(-1)
xu = F.unfold(x8.float(), (3, 3), padding=1).view(2, 8, 9, -1)
acc = torch.zeros(2, 8, xu.shape[-1], device=dev)
for t in range(9):
    acc = acc + lutf[(xu[:, :, t] + 128).long() * 256 + (w[:, t] + 128).long().view(1, 8, 1)]
report(torch.equal(y1.flatten(2), acc), 'F int16-boundary LUT bitwise')
lutb = lut_bad.reshape(-1)
acc2 = torch.zeros_like(acc)
for t in range(9):
    acc2 = acc2 + lutb[(xu[:, :, t] + 128).long() * 256 + (w[:, t] + 128).long().view(1, 8, 1)]
report(torch.equal(y2.flatten(2), acc2), 'F fp32-fallback LUT bitwise')

lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
for g, Cc in ((1, 16), (16, 16)):
    m = make_module(Cc, 16, 3, 1, 1, 1, g, 'lre', lut)
    xbase = torch.randn(4, Cc, 14, 14, device=dev)
    y_c = m(xbase)
    y_cl = m(xbase.to(memory_format=torch.channels_last))
    y_sl = m(torch.cat([xbase, xbase], 0)[4:])     # storage-offset view
    report(torch.equal(y_c, y_cl) and torch.equal(y_c, y_sl),
           f'F non-contiguous inputs (g={g})')

# ---- G. end-to-end: swapped MobileNetV2 overfits one batch --------------
print('--- G. MobileNetV2 (width 0.35) overfit-one-batch ---')
from torchvision.models import mobilenet_v2
for mode in ('lre', 'ste'):
    torch.manual_seed(1)
    net = mobilenet_v2(width_mult=0.35, num_classes=10)
    def swap(mod):
        for name, ch in mod.named_children():
            if isinstance(ch, nn.Conv2d):
                q = Conv2d_int8(ch.in_channels, ch.out_channels, ch.kernel_size,
                                exact_lut, grad=mode, dx=id_table, dw=id_table,
                                bias=ch.bias is not None, stride=ch.stride,
                                padding=ch.padding, dilation=ch.dilation,
                                groups=ch.groups)
                with torch.no_grad():
                    q.weight.copy_(ch.weight)
                setattr(mod, name, q)
            else:
                swap(ch)
    swap(net)
    net = net.to(dev).train()
    xb = torch.randn(16, 3, 96, 96, device=dev)
    yb = torch.randint(0, 10, (16,), device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    l0 = lN = None
    for i in range(150):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(xb), yb)
        loss.backward()
        opt.step()
        if i == 0:
            l0 = float(loss.detach())
        lN = float(loss.detach())
    report(lN < 0.1 * l0 and lN < 0.5,
           f'G {mode}: loss {l0:.3f} -> {lN:.4f} in 150 steps')

print('\nALL PASS' if nfail == 0 else f'\n{nfail} FAILURES')
raise SystemExit(0 if nfail == 0 else 1)
