# Stage-by-stage profile of Conv2d_int8_decoupled (lre) training step.
# Replicates the module's forward internals so each stage can be timed with
# CUDA events; backward pieces are timed by calling the ops directly.
import torch
import torch.nn.functional as F
import approxtorch as at
from approxtorch.nn.Conv2d_int8_decoupled import Conv2d_int8

torch.manual_seed(0)
dev = 'cuda'

def evtime(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times)//2]

shapes = [
    # (B, C, H, O, k, stride, pad)  ResNet-50-ish
    (64, 64, 56, 64, 3, 1, 1),
    (64, 256, 14, 256, 3, 1, 1),
    (64, 64, 56, 256, 1, 1, 0),
    (128, 16, 32, 16, 3, 1, 1),   # CIFAR ResNet20-ish
]

lut = torch.randint(-127*127, 127*127, (256, 256), device=dev).float()
dx = torch.randn(256, device=dev)
dw = torch.randn(256, device=dev)

for (B, C, H, O, k, s, p) in shapes:
    print(f'\n=== B{B} C{C} H{H} O{O} k{k} s{s} p{p} ===')
    m = Conv2d_int8(C, O, k, lut, grad='lre', dx=dx, dw=dw, bias=True,
                    stride=s, padding=p).to(dev)
    m.train()
    x0 = torch.randn(B, C, H, H, device=dev)
    OH = (H + 2*p - k)//s + 1
    L = OH*OH
    K = C*k*k

    # ---- full step timing ----
    def step():
        x = x0.detach().requires_grad_(True)
        y = m(x)
        y.backward(torch.ones_like(y))
    t_full = evtime(step, iters=20)

    # forward-only
    def fwd():
        x = x0.detach().requires_grad_(True)
        return m(x)
    t_fwd = evtime(lambda: fwd(), iters=20)
    print(f'full step {t_full:8.3f} ms   fwd {t_fwd:8.3f} ms   bwd ~{t_full-t_fwd:8.3f} ms')

    # held memory after forward
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    x = x0.detach().requires_grad_(True)
    y = m(x)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    del y; del x
    print(f'held after fwd {held/2**20:7.1f} MB   peak during fwd {peak/2**20:7.1f} MB')

    # ---- forward stages ----
    with torch.no_grad():
        t = {}
        t['0 absmax (x.abs().max)'] = evtime(lambda: x0.abs().max())
        t['0b vector_norm inf    '] = evtime(lambda: torch.linalg.vector_norm(x0, torch.inf))
        xq = at.nn.fakequant.symmetric_static_quantize_int8_per_tensor(x0, m.scale_x, None, -127, 127)
        t['1 fakequant x (fused) '] = evtime(lambda: torch.ops.approxtorch.fakequant_per_tensor_claude.default(x0, m.scale_x, -127, 127))
        t['2 fakequant w (python)'] = evtime(lambda: at.nn.fakequant.symmetric_dynamic_quantize_int8_per_channel(m.weight, ch_axis=0))
        t['3 xq_pre cast .to(i8) '] = evtime(lambda: xq.detach().to(torch.int8))
        if k != 1:
            t['4 unfold fp32         '] = evtime(lambda: F.unfold(xq, (k, k), padding=p, stride=s))
            xi8 = xq.to(torch.int8)
            try:
                F.unfold(xi8, (k, k), padding=p, stride=s)
                t['4b unfold int8        '] = evtime(lambda: F.unfold(xi8, (k, k), padding=p, stride=s))
            except Exception as ex:
                print('   unfold int8 unsupported:', type(ex).__name__, str(ex)[:80])
        xu = F.unfold(xq, (k, k), padding=p, stride=s) if k != 1 else xq.flatten(2)
        wq, sw = at.nn.fakequant.symmetric_dynamic_quantize_int8_per_channel(m.weight, ch_axis=0)
        wf = wq.view(O, -1)
        t['5 bgemm fwd (no save) '] = evtime(lambda: at.backend.ops.bgemm_fake_int8_claude(xu, wf, lut))
        t['5b bgemm fwd (_save)  '] = evtime(lambda: at.backend.ops.bgemm_fake_int8_claude_save(xu, wf, lut))
        y_raw = at.backend.ops.bgemm_fake_int8_claude(xu, wf, lut).view(B, O, OH, OH)
        sv = (m.scale_x * sw).view(1, -1, 1, 1)
        t['6 dequant addcmul     '] = evtime(lambda: torch.addcmul(m.bias.view(1, -1, 1, 1), y_raw, sv))

        # ---- backward stages ----
        go = torch.ones(B, O, L, device=dev)
        xi8_img = xq.to(torch.int8)
        _, xq_u8, wq_u8 = at.backend.ops.bgemm_fake_int8_claude_save(xu, wf, lut)
        wT = wq_u8.transpose(0, 1).contiguous()
        t['7 lre bwd im2col op   '] = evtime(lambda: at.backend.ops.bgemm_lre_backward_claude_im2col(
            go, xi8_img, wT, dx, dw, (k, k), (s, s), (p, p), (1, 1)))
        gx_unf, gw = at.backend.ops.bgemm_lre_backward_claude_im2col(
            go, xi8_img, wT, dx, dw, (k, k), (s, s), (p, p), (1, 1))
        if k != 1:
            t['8 fold (unfold bwd)   '] = evtime(lambda: F.fold(gx_unf, (H, H), (k, k), padding=p, stride=s))
        mask = torch.ops.approxtorch.fakequant_per_tensor_claude.default(x0, m.scale_x, -127, 127)[1]
        gimg = torch.randn(B, C, H, H, device=dev)
        t['9 fakequant x bwd     '] = evtime(lambda: torch.ops.approxtorch.fakequant_per_tensor_backward_claude.default(gimg, mask, m.scale_x))

    tot = sum(v for kk, v in t.items() if not kk.startswith('0b') and not kk.startswith('4b') and not kk.startswith('5 '))
    for kk, v in t.items():
        print(f'  {kk} {v:8.3f} ms')
    print(f'  (sum of used stages ~ {tot:.3f} ms; KNL fp32 = {B*K*L*4/2**20:.0f} MB)')
