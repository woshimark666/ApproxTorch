# Accuracy tests for the restructured Conv2d_int8_decoupled lre path:
#   forward: int8 image -> im2col_u8 -> LUT-BGEMM (u8 input, no fp32 unfold,
#            no prepass) -- must be BIT-IDENTICAL to the old pipeline
#   backward: cuDNN convolution_backward on LUT-mapped operands (k != 1) /
#            cuBLAS op (1x1) -- must match the old pipeline to fp32 round-off
#            and the fp64 truth to ~1e-6 relative error.
#
# The "old pipeline" is rebuilt here out of the still-present building blocks
# (fakequant fns + F.unfold + the gemm-level _bgemm_int8_lre Function with an
# explicit fold), so old and new run on identical quantized inputs.
import torch
import torch.nn.functional as F
from torch.autograd import Function
import approxtorch as at
from approxtorch.nn import fakequant, bgemm
from approxtorch.nn.Conv2d_int8_decoupled import Conv2d_int8

torch.manual_seed(0)
dev = 'cuda'
nfail = 0


def relerr(a, b):
    b = b.double()
    return ((a.double() - b).norm() / (b.norm() + 1e-30)).item()


# old conv path: fp32 unfold (autograd) + gemm-level Function.
# lre: claude gemm-level op (bit-identical to pre-change module by its own
# op tests). ste: the original einsum-backward Function saving fp32.
def old_path(m, x):
    B = x.shape[0]
    O, C, kH, kW = m.weight.shape
    xq = fakequant.symmetric_static_quantize_int8_per_tensor(
        x, m.scale_x, None, m.qmin, m.qmax)
    w, s_w = fakequant.symmetric_dynamic_quantize_int8_per_channel(
        m.weight, ch_axis=0, bits=m.weight_bits)
    xu = F.unfold(xq, m.kernel_size, dilation=m.dilation,
                  padding=m.padding, stride=m.stride)
    if m.grad == 'ste':
        y = bgemm.bgemm_int8_ste(xu, w.view(O, -1), m.lut)
    else:
        y = bgemm.bgemm_int8_lre(xu, w.view(O, -1), m.lut, m.dx, m.dw)
    H, W = x.shape[2], x.shape[3]
    OH = (H + 2*m.padding[0] - m.dilation[0]*(kH-1) - 1)//m.stride[0] + 1
    OW = (W + 2*m.padding[1] - m.dilation[1]*(kW-1) - 1)//m.stride[1] + 1
    y = y.view(B, O, OH, OW)
    s = (m.scale_x * s_w).view(1, -1, 1, 1)
    if m.bias is not None:
        y = torch.addcmul(m.bias.view(1, -1, 1, 1), y, s)
    else:
        y = y * s
    return y


# fp64 truth for the module-level gradients, replicating the whole chain
# (fakequant STE masks included) with float64 LUT math.
def fp64_grads(m, x, go):
    with torch.no_grad():
        O, C, kH, kW = m.weight.shape
        B, _, H, W = x.shape
        # quantization replicated in fp32 (the kernels divide/round in fp32;
        # fp64 division could land in a different integer bucket at .5
        # boundaries) -- only the gradient math below runs in fp64
        scale_x = m.scale_x
        xs = x / scale_x
        xmask = (xs >= m.qmin) & (xs <= m.qmax)
        xq = torch.clamp(torch.round(xs), m.qmin, m.qmax)
        qmaxw = 2 ** (m.weight_bits - 1) - 1
        absmax = m.weight.detach().abs().amax(dim=(1, 2, 3))
        s_w = (absmax / qmaxw).clamp(min=1e-12)
        ws = m.weight / s_w.view(-1, 1, 1, 1)
        wmask = (ws >= -qmaxw) & (ws <= qmaxw)
        wq = torch.clamp(torch.round(ws), -qmaxw, qmaxw)
        scale_x = scale_x.double()
        s_w = s_w.double()
        # unfold
        xu = F.unfold(xq, m.kernel_size, dilation=m.dilation,
                      padding=m.padding, stride=m.stride).double()  # [B,K,L]
        wf = wq.view(O, -1)                                          # [O,K]
        OH = (H + 2*m.padding[0] - m.dilation[0]*(kH-1) - 1)//m.stride[0] + 1
        OW = (W + 2*m.padding[1] - m.dilation[1]*(kW-1) - 1)//m.stride[1] + 1
        L = OH * OW
        # dequant scale
        s = (scale_x * s_w).view(1, O, 1, 1)
        go64 = go.double()
        if m.bias is not None:
            g_bias = go64.sum(dim=(0, 2, 3))
        else:
            g_bias = None
        g_raw = (go64 * s).view(B, O, L)                # grad wrt bgemm output
        # lre backward in fp64 (ste == identity LUTs: DX[q]=q, DW[q]=q)
        if m.grad == 'ste':
            dxl = (torch.arange(256, device=x.device) - 128).double()
            dwl = dxl
        else:
            dxl, dwl = m.dx.double(), m.dw.double()
        xidx = (xu + 128).long()
        widx = (wf + 128).long()
        wp = dxl[widx]                                  # [O,K]
        xp = dwl[xidx]                                  # [B,K,L]
        gx_u = torch.einsum('ok,nol->nkl', wp, g_raw)
        gw = torch.einsum('nol,nkl->ok', g_raw, xp).view(O, C, kH, kW)
        # fold + fakequant STE
        gx = F.fold(gx_u.float(), (H, W), m.kernel_size, dilation=m.dilation,
                    padding=m.padding, stride=m.stride).double()
        gx = gx * xmask / scale_x
        gw = gw * wmask / s_w.view(-1, 1, 1, 1)
        return gx, gw, g_bias


def check(tag, B, C, H, W, O, k, s=1, p=0, d=1, bias=True, train=True,
          mode='lre'):
    global nfail
    lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
    dx = torch.randn(256, device=dev) if mode == 'lre' else None
    dw = torch.randn(256, device=dev) if mode == 'lre' else None
    m = Conv2d_int8(C, O, k, lut, grad=mode, dx=dx, dw=dw, bias=bias,
                    stride=s, padding=p, dilation=d).to(dev)
    m.train(train)
    m.update_scale = False
    with torch.no_grad():
        m.scale_x.fill_(0.02)
    x0 = torch.randn(B, C, H, W, device=dev) * 2.0
    with torch.no_grad():
        go = torch.randn_like(m(torch.randn(B, C, H, W, device=dev)))

    # ---- new path ----
    x = x0.detach().requires_grad_(True)
    m.weight.grad = None
    if m.bias is not None:
        m.bias.grad = None
    y_new = m(x)
    y_new.backward(go)
    gx_new, gw_new = x.grad.clone(), m.weight.grad.clone()
    gb_new = m.bias.grad.clone() if m.bias is not None else None

    # ---- old path ----
    x = x0.detach().requires_grad_(True)
    m.weight.grad = None
    if m.bias is not None:
        m.bias.grad = None
    y_old = old_path(m, x)
    y_old.backward(go)
    gx_old, gw_old = x.grad.clone(), m.weight.grad.clone()
    gb_old = m.bias.grad.clone() if m.bias is not None else None

    # ---- fp64 truth ----
    gx_t, gw_t, gb_t = fp64_grads(m, x0, go)

    bit = torch.equal(y_new, y_old)
    ex_n, ex_o = relerr(gx_new, gx_t), relerr(gx_old, gx_t)
    ew_n, ew_o = relerr(gw_new, gw_t), relerr(gw_old, gw_t)
    ok = bit and ex_n < 2e-6 and ew_n < 2e-6
    if gb_new is not None:
        eb_n = relerr(gb_new, gb_t)
        ok = ok and eb_n < 2e-6
    else:
        eb_n = float('nan')
    if not ok:
        nfail += 1
    print(f'{"PASS" if ok else "FAIL"} {tag:34s} y bit-identical={bit}  '
          f'gx {ex_n:.2e} (old {ex_o:.2e})  gw {ew_n:.2e} (old {ew_o:.2e})  gb {eb_n:.2e}')


# component-level: im2col_u8 vs quantized fp32 unfold must be bit-identical
def check_im2col(B, C, H, W, k, s, p, d):
    global nfail
    x8 = torch.randint(-127, 128, (B, C, H, W), device=dev, dtype=torch.int8)
    ref = (F.unfold(x8.float(), (k, k), dilation=d, padding=p, stride=s)
           + 128).to(torch.uint8)
    got = at.backend.ops.im2col_u8(x8, (k, k), (s, s), (p, p), (d, d))
    ok = torch.equal(got, ref)
    if not ok:
        nfail += 1
    print(f'{"PASS" if ok else "FAIL"} im2col_u8 B{B} C{C} {H}x{W} k{k} s{s} p{p} d{d}')


def check_maps():
    global nfail
    lut = torch.randn(256, device=dev)
    w8 = torch.randint(0, 256, (37, 129), device=dev, dtype=torch.uint8)
    ok1 = torch.equal(at.backend.ops.lut_map(w8, lut), lut[w8.long()])
    x8 = torch.randint(-127, 128, (3, 5, 9, 11), device=dev, dtype=torch.int8)
    ref = F.pad(lut[(x8.long() + 128)], (2, 2, 1, 1), value=float(lut[128]))
    ok2 = torch.equal(at.backend.ops.lut_map_pad(x8, lut, (1, 2)), ref)
    if not (ok1 and ok2):
        nfail += 1
    print(f'{"PASS" if ok1 and ok2 else "FAIL"} lut_map / lut_map_pad exact')


check_maps()
check_im2col(2, 3, 17, 19, 3, 1, 1, 1)
check_im2col(2, 4, 16, 16, 3, 2, 1, 1)
check_im2col(1, 2, 9, 9, 5, 1, 2, 2)
check_im2col(3, 2, 8, 8, 1, 2, 0, 1)

check('k3 s1 p1 resnet block',   8, 64, 28, 28, 64, 3, 1, 1)
check('k3 s2 p1 downsample',     8, 64, 28, 28, 128, 3, 2, 1)
check('k1 s1 p0 bottleneck',     8, 64, 28, 28, 256, 1, 1, 0)
check('k1 s2 p0 shortcut',       8, 64, 28, 28, 128, 1, 2, 0)
check('k7 s2 p3 stem C=3',       4, 3, 64, 64, 32, 7, 2, 3)
check('k5 s1 p2',                4, 16, 20, 20, 24, 5, 1, 2)
check('k3 s1 p2 d2 dilated',     4, 16, 20, 20, 24, 3, 1, 2, d=2)
check('k3 s1 p0 no-pad',         4, 16, 20, 20, 24, 3, 1, 0)
check('k3 s1 p1 no bias',        4, 16, 20, 20, 24, 3, 1, 1, bias=False)
check('k3 s1 p1 batch1',         1, 16, 20, 20, 24, 3, 1, 1)
check('k1 s1 p0 batch1',         1, 16, 20, 20, 24, 1, 1, 0)
check('k3 s1 p1 H!=W',           4, 16, 24, 18, 24, 3, 1, 1)
check('k3 s1 p1 cifar tiny',     16, 16, 32, 32, 16, 3, 1, 1)

print('--- ste (backward == plain conv backward on quantized values) ---')
check('ste k3 s1 p1 resnet block', 8, 64, 28, 28, 64, 3, 1, 1, mode='ste')
check('ste k3 s2 p1 downsample',   8, 64, 28, 28, 128, 3, 2, 1, mode='ste')
check('ste k1 s1 p0 bottleneck',   8, 64, 28, 28, 256, 1, 1, 0, mode='ste')
check('ste k1 s2 p0 shortcut',     8, 64, 28, 28, 128, 1, 2, 0, mode='ste')
check('ste k7 s2 p3 stem C=3',     4, 3, 64, 64, 32, 7, 2, 3, mode='ste')
check('ste k3 s1 p2 d2 dilated',   4, 16, 20, 20, 24, 3, 1, 2, d=2, mode='ste')
check('ste k3 s1 p1 no bias',      4, 16, 20, 20, 24, 3, 1, 1, bias=False, mode='ste')
check('ste k3 s1 p1 batch1',       1, 16, 20, 20, 24, 3, 1, 1, mode='ste')
check('ste k3 s1 p1 H!=W',         4, 16, 24, 18, 24, 3, 1, 1, mode='ste')

# ---------------- depthwise (groups == C == O) ----------------

# manual tap-ordered LUT reference: same ascending (i,j) accumulation order
# as both the dwconv kernel and a per-channel im2col GEMM -> bit-comparable
def dw_ref_y(xq, w_vals, lut, kernel_size, stride, padding, dilation):
    B, C, H, W = xq.shape
    kH, kW = kernel_size
    kk = kH * kW
    xu = F.unfold(xq, kernel_size, dilation=dilation, padding=padding,
                  stride=stride).view(B, C, kk, -1)            # [B,C,kk,L]
    xidx = (xu + 128).long()
    widx = (w_vals + 128).long()                               # [C,kk]
    lutf = lut.reshape(-1)
    acc = torch.zeros(B, C, xu.shape[-1], device=xq.device)
    for t in range(kk):
        acc = acc + lutf[xidx[:, :, t, :] * 256 + widx[:, t].view(1, C, 1)]
    return acc                                                  # [B,C,L]


def check_dwconv_op(B, C, H, W, k, s, p, d, int_lut=True):
    global nfail
    if int_lut:
        lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
    else:
        lut = torch.randn(256, 256, device=dev)  # non-integer -> fp32 LUT path
    x8 = torch.randint(-127, 128, (B, C, H, W), device=dev, dtype=torch.int8)
    w = torch.randint(-127, 128, (C, k * k), device=dev).float()
    y, wq = at.backend.ops.dwconv_fake_int8_claude(
        x8, w, lut, (k, k), (s, s), (p, p), (d, d))
    OH = (H + 2*p - d*(k-1) - 1)//s + 1
    OW = (W + 2*p - d*(k-1) - 1)//s + 1
    ref = dw_ref_y(x8.float(), w, lut, (k, k), (s, s), (p, p), (d, d))
    ok = torch.equal(y.view(B, C, -1), ref) \
        and torch.equal(wq, (w + 128).to(torch.uint8)) \
        and y.shape == (B, C, OH, OW)
    if not ok:
        nfail += 1
    print(f'{"PASS" if ok else "FAIL"} dwconv op B{B} C{C} {H}x{W} k{k} s{s} p{p} d{d}'
          f' {"i16lut" if int_lut else "f32lut"}')


# fp64 truth for depthwise module grads (grouped einsums)
def fp64_grads_dw(m, x, go):
    with torch.no_grad():
        C = m.in_channels
        B, _, H, W = x.shape
        kH, kW = m.kernel_size
        kk = kH * kW
        scale_x = m.scale_x
        xs = x / scale_x
        xmask = (xs >= m.qmin) & (xs <= m.qmax)
        xq = torch.clamp(torch.round(xs), m.qmin, m.qmax)
        qmaxw = 2 ** (m.weight_bits - 1) - 1
        absmax = m.weight.detach().abs().amax(dim=(1, 2, 3))
        s_w = (absmax / qmaxw).clamp(min=1e-12)
        ws = m.weight / s_w.view(-1, 1, 1, 1)
        wmask = (ws >= -qmaxw) & (ws <= qmaxw)
        wq = torch.clamp(torch.round(ws), -qmaxw, qmaxw)

        xu = F.unfold(xq, m.kernel_size, dilation=m.dilation, padding=m.padding,
                      stride=m.stride).view(B, C, kk, -1).double()  # [B,C,kk,L]
        wf = wq.view(C, kk).double()
        if m.grad == 'ste':
            dxl = (torch.arange(256, device=x.device) - 128).double()
            dwl = dxl
        else:
            dxl, dwl = m.dx.double(), m.dw.double()
        s = (scale_x * s_w).double().view(1, C, 1, 1)
        go64 = go.double()
        g_bias = go64.sum(dim=(0, 2, 3)) if m.bias is not None else None
        g_raw = (go64 * s).flatten(2)                       # [B,C,L]
        wp = dxl[(wf + 128).long()]                         # [C,kk]
        xp = dwl[(xu + 128).long()]                         # [B,C,kk,L]
        gx_u = torch.einsum('bcl,ct->bctl', g_raw, wp).reshape(B, C * kk, -1)
        gx = F.fold(gx_u.float(), (H, W), m.kernel_size, dilation=m.dilation,
                    padding=m.padding, stride=m.stride).double()
        gx = gx * xmask / scale_x.double()
        gw = torch.einsum('bcl,bctl->ct', g_raw, xp).view(C, 1, kH, kW)
        gw = gw * wmask / s_w.double().view(-1, 1, 1, 1)
        return gx, gw, g_bias


def check_dw(tag, B, C, H, W, k, s=1, p=0, d=1, bias=True, mode='lre'):
    global nfail
    lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
    dx = torch.randn(256, device=dev) if mode == 'lre' else None
    dw = torch.randn(256, device=dev) if mode == 'lre' else None
    m = Conv2d_int8(C, C, k, lut, grad=mode, dx=dx, dw=dw, bias=bias,
                    stride=s, padding=p, dilation=d, groups=C).to(dev)
    m.train()
    m.update_scale = False
    with torch.no_grad():
        m.scale_x.fill_(0.02)
    x0 = torch.randn(B, C, H, W, device=dev) * 2.0
    with torch.no_grad():
        go = torch.randn_like(m(torch.randn(B, C, H, W, device=dev)))

    x = x0.detach().requires_grad_(True)
    m.weight.grad = None
    if m.bias is not None:
        m.bias.grad = None
    y_new = m(x)
    y_new.backward(go)

    # forward reference through the same python fakequant chain
    with torch.no_grad():
        xqr = fakequant.symmetric_static_quantize_int8_per_tensor(
            x0, m.scale_x, None, m.qmin, m.qmax)
        wr, s_wr = fakequant.symmetric_dynamic_quantize_int8_per_channel(
            m.weight, ch_axis=0, bits=m.weight_bits)
        yr = dw_ref_y(xqr, wr.view(C, -1), m.lut, m.kernel_size,
                      m.stride, m.padding, m.dilation)
        yr = yr.view_as(y_new)
        sr = (m.scale_x * s_wr).view(1, -1, 1, 1)
        if m.bias is not None:
            yr = torch.addcmul(m.bias.view(1, -1, 1, 1), yr, sr)
        else:
            yr = yr * sr
    bit = torch.equal(y_new, yr)

    gx_t, gw_t, gb_t = fp64_grads_dw(m, x0, go)
    ex = relerr(x.grad, gx_t)
    ew = relerr(m.weight.grad, gw_t)
    eb = relerr(m.bias.grad, gb_t) if m.bias is not None else float('nan')
    ok = bit and ex < 2e-6 and ew < 2e-6 and (m.bias is None or eb < 2e-6)
    if not ok:
        nfail += 1
    print(f'{"PASS" if ok else "FAIL"} {tag:34s} y bit-identical={bit}  '
          f'gx {ex:.2e}  gw {ew:.2e}  gb {eb:.2e}')


print('--- depthwise op (forward bit-exact vs tap-ordered reference) ---')
check_dwconv_op(2, 8, 17, 19, 3, 1, 1, 1)
check_dwconv_op(2, 16, 16, 16, 3, 2, 1, 1)
check_dwconv_op(1, 4, 12, 12, 5, 1, 2, 1)        # generic-k template path
check_dwconv_op(2, 8, 13, 11, 3, 1, 2, 2)        # dilated
check_dwconv_op(2, 8, 9, 9, 3, 1, 0, 1)          # no pad
check_dwconv_op(2, 8, 16, 16, 3, 1, 1, 1, int_lut=False)  # fp32-LUT path
check_dwconv_op(1, 960, 7, 7, 3, 1, 1, 1)        # mobilenet last dw shape

print('--- depthwise module (lre + ste) ---')
check_dw('dw lre k3 s1 p1 C32',  8, 32, 28, 28, 3, 1, 1)
check_dw('dw lre k3 s2 p1 C96',  8, 96, 14, 14, 3, 2, 1)
check_dw('dw lre k5 s1 p2 C16',  4, 16, 20, 20, 5, 1, 2)
check_dw('dw lre k3 s1 p1 batch1', 1, 32, 14, 14, 3, 1, 1)
check_dw('dw lre k3 s1 p1 no bias', 4, 16, 16, 16, 3, 1, 1, bias=False)
check_dw('dw ste k3 s1 p1 C32',  8, 32, 28, 28, 3, 1, 1, mode='ste')
check_dw('dw ste k3 s2 p1 C96',  8, 96, 14, 14, 3, 2, 1, mode='ste')
check_dw('dw ste k5 s1 p2 C16',  4, 16, 20, 20, 5, 1, 2, mode='ste')
check_dw('dw ste k3 s1 p1 H!=W', 4, 24, 20, 14, 3, 1, 1, mode='ste')

# eval mode / no-grad forward still works and matches
m = Conv2d_int8(16, 24, 3, torch.randint(-127*127, 127*127, (256, 256), device=dev).float(),
                grad='lre', dx=torch.randn(256, device=dev),
                dw=torch.randn(256, device=dev), bias=True, padding=1).to(dev)
m.eval()
with torch.no_grad():
    xe = torch.randn(4, 16, 20, 20, device=dev)
    ye = m(xe)
    yo = old_path(m, xe)
ok = torch.equal(ye, yo)
if not ok:
    nfail += 1
print(f'{"PASS" if ok else "FAIL"} eval/no-grad forward bit-identical')

print('\nALL PASS' if nfail == 0 else f'\n{nfail} FAILURES')
raise SystemExit(0 if nfail == 0 else 1)
