# Depthwise LUT conv benchmark on MobileNetV2 (width 1.0) depthwise layers,
# plus a whole-network smoke test: torchvision mobilenet_v2 with every conv
# swapped to Conv2d_int8 (one fwd+bwd training step).
#
#   fwd column:   dwconv_fake_int8_claude alone
#   cudnn fp32:   F.conv2d depthwise fp32 (speed-of-light reference; it does
#                 1 MAC where we do 1 LUT gather per tap)
#   step ste/lre: full module fwd+bwd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import approxtorch as at
from approxtorch.nn.Conv2d_int8_decoupled import Conv2d_int8

torch.manual_seed(0)
dev = 'cuda'
B = 64

def ev(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]

lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
dx = torch.randn(256, device=dev)
dwl = torch.randn(256, device=dev)

# (C, H, stride) of all distinct MobileNetV2 depthwise layers, width 1.0, 224 input
layers = [(32, 112, 1), (96, 112, 2), (144, 56, 1), (144, 56, 2),
          (192, 28, 1), (192, 28, 2), (384, 14, 1), (576, 14, 1),
          (576, 14, 2), (960, 7, 1)]

print(f'{"layer":20s} {"fwd op":>8s} {"cudnn fp32":>10s} {"step ste":>9s} {"step lre":>9s}')
for (C, H, s) in layers:
    x8 = torch.randint(-127, 128, (B, C, H, H), device=dev, dtype=torch.int8)
    w = torch.randint(-127, 128, (C, 9), device=dev).float()
    t_op = ev(lambda: at.backend.ops.dwconv_fake_int8_claude(
        x8, w, lut, (3, 3), (s, s), (1, 1), (1, 1)))
    xf = x8.float()
    wf = w.view(C, 1, 3, 3)
    torch.backends.cudnn.benchmark = True
    t_ref = ev(lambda: F.conv2d(xf, wf, None, (s, s), (1, 1), 1, C))

    steps = {}
    for mode in ('ste', 'lre'):
        kw = dict(dx=dx, dw=dwl) if mode == 'lre' else {}
        m = Conv2d_int8(C, C, 3, lut, grad=mode, bias=True, stride=s,
                        padding=1, groups=C, **kw).to(dev)
        m.train(); m.update_scale = False
        with torch.no_grad():
            m.scale_x.fill_(0.02)
        x0 = torch.randn(B, C, H, H, device=dev)
        def step():
            xx = x0.detach().requires_grad_(True)
            y = m(xx)
            y.backward(torch.ones_like(y))
        steps[mode] = ev(step, iters=20)
    print(f'C{C} {H}x{H} s{s}'.ljust(20) +
          f' {t_op:8.3f} {t_ref:10.3f} {steps["ste"]:9.3f} {steps["lre"]:9.3f}')

# ---------------- MobileNetV2 end-to-end smoke test ----------------
print('\nMobileNetV2 full-network swap (lre), one training step:')
from torchvision.models import mobilenet_v2
net = mobilenet_v2(weights=None)

def swap(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            q = Conv2d_int8(child.in_channels, child.out_channels,
                            child.kernel_size, lut, grad='lre', dx=dx, dw=dwl,
                            bias=child.bias is not None, stride=child.stride,
                            padding=child.padding, dilation=child.dilation,
                            groups=child.groups)
            with torch.no_grad():
                q.weight.copy_(child.weight)
                if child.bias is not None:
                    q.bias.copy_(child.bias)
            setattr(module, name, q)
        else:
            swap(child)

swap(net)
net = net.to(dev).train()
n_conv = sum(1 for mm in net.modules() if isinstance(mm, Conv2d_int8))
n_dw = sum(1 for mm in net.modules() if isinstance(mm, Conv2d_int8) and mm.groups > 1)
print(f'  swapped convs: {n_conv} ({n_dw} depthwise)')

xb = torch.randn(B, 3, 224, 224, device=dev)
yb = torch.randint(0, 1000, (B,), device=dev)
opt = torch.optim.SGD(net.parameters(), lr=1e-3)

def train_step():
    opt.zero_grad(set_to_none=True)
    out = net(xb)
    loss = F.cross_entropy(out, yb)
    loss.backward()
    opt.step()
    return loss.detach()

loss = train_step()
torch.cuda.synchronize()
print(f'  one step OK, loss = {float(loss):.4f}')
t = ev(train_step, iters=10, warmup=5)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
train_step(); torch.cuda.synchronize()
print(f'  step time {t:.1f} ms (B={B}, 224x224), peak mem '
      f'{torch.cuda.max_memory_allocated()/2**30:.2f} GB')
