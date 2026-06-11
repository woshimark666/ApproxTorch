# Interleaved A/B benchmark: restructured Conv2d_int8_decoupled lre path vs
# the old pipeline (fp32 unfold + gemm-level Function), same process, same
# tensors. Reports step time (fwd+bwd) and forward memory.
import torch
import torch.nn.functional as F
from approxtorch.nn import fakequant, bgemm
from approxtorch.nn.Conv2d_int8_decoupled import Conv2d_int8

torch.manual_seed(0)
dev = 'cuda'


def old_path(m, x):
    B = x.shape[0]
    O, C, kH, kW = m.weight.shape
    xq = fakequant.symmetric_static_quantize_int8_per_tensor(
        x, m.scale_x, None, m.qmin, m.qmax)
    w, s_w = fakequant.symmetric_dynamic_quantize_int8_per_channel(
        m.weight, ch_axis=0, bits=m.weight_bits)
    if m.kernel_size == (1, 1) and m.padding == (0, 0) and m.stride == (1, 1):
        xu = xq.flatten(2)
    else:
        xu = F.unfold(xq, m.kernel_size, dilation=m.dilation,
                      padding=m.padding, stride=m.stride)
    y = bgemm.bgemm_int8_lre(xu, w.view(O, -1), m.lut, m.dx, m.dw)
    H, W = x.shape[2], x.shape[3]
    OH = (H + 2*m.padding[0] - m.dilation[0]*(kH-1) - 1)//m.stride[0] + 1
    OW = (W + 2*m.padding[1] - m.dilation[1]*(kW-1) - 1)//m.stride[1] + 1
    y = y.view(B, O, OH, OW)
    s = (m.scale_x * s_w).view(1, -1, 1, 1)
    return torch.addcmul(m.bias.view(1, -1, 1, 1), y, s)


def step_time(fn, iters):
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


shapes = [
    (64, 64, 56, 64, 3, 1, 1),
    (64, 256, 14, 256, 3, 1, 1),
    (64, 64, 56, 256, 1, 1, 0),
    (128, 16, 32, 16, 3, 1, 1),
    (64, 512, 7, 512, 3, 1, 1),
    (32, 128, 28, 128, 3, 2, 1),
]

lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
dx = torch.randn(256, device=dev)
dw = torch.randn(256, device=dev)

print(f'{"shape":28s} {"old ms":>8s} {"new ms":>8s} {"speedup":>8s}'
      f' {"old peak":>9s} {"new peak":>9s}')
for (B, C, H, O, k, s, p) in shapes:
    m = Conv2d_int8(C, O, k, lut, grad='lre', dx=dx, dw=dw, bias=True,
                    stride=s, padding=p).to(dev)
    m.train()
    m.update_scale = False
    with torch.no_grad():
        m.scale_x.fill_(0.02)
    x0 = torch.randn(B, C, H, H, device=dev)

    def new_step():
        x = x0.detach().requires_grad_(True)
        y = m(x)
        y.backward(torch.ones_like(y))

    def old_step():
        x = x0.detach().requires_grad_(True)
        y = old_path(m, x)
        y.backward(torch.ones_like(y))

    for _ in range(8):
        new_step(); old_step()
    torch.cuda.synchronize()
    t_new, t_old = [], []
    for _ in range(6):  # interleave batches to ride out clock drift
        t_new += step_time(new_step, 5)
        t_old += step_time(old_step, 5)
    t_new.sort(); t_old.sort()
    mn, mo = t_new[len(t_new)//2], t_old[len(t_old)//2]

    def peak(fn):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 2**20

    pk_new, pk_old = peak(new_step), peak(old_step)
    tag = f'B{B} C{C} H{H} O{O} k{k} s{s} p{p}'
    print(f'{tag:28s} {mo:8.3f} {mn:8.3f} {mo/mn:7.2f}x {pk_old:8.1f}M {pk_new:8.1f}M')
