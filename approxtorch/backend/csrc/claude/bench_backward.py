#!/usr/bin/env python
"""Correctness + performance harness for the claude-optimized LRE backward.

JIT-compiles the reference (cuda/bgemm_lre_backward.cu) and the optimized
op (claude/bgemm_lre_backward_claude.cu) and compares both against an exact
fp64 ground truth:

  grad_x[n,k,l] = sum_o go[n,o,l] * DX[q(w[k,o])]
  grad_w[k,o]   = sum_{n,l} go[n,o,l] * DW[q(x[n,k,l])]

The optimized op accumulates via cuBLAS, so it matches the reference to fp32
round-off (allclose), not bit-exactly.

Usage:
  python bench_backward.py            # correctness + benchmark (auto cfg)
  python bench_backward.py --sweep    # also time both grad_w strategies
  python bench_backward.py --quick    # fewer iters
"""
import argparse
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.join(HERE, "..", "cuda")

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


def build():
    load(
        name="lre_backward_ref",
        sources=[os.path.join(CUDA_DIR, "bgemm_lre_backward.cu")],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    load(
        name="lre_backward_claude",
        sources=[os.path.join(HERE, "bgemm_lre_backward_claude.cu")],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )


# (N, K, L, O) — x:[N,K,L], w:[K,O], grad_output:[N,O,L]
SHAPES = [
    # CIFAR ResNet20-ish
    (128, 27, 1024, 16),
    (128, 144, 1024, 16),
    (128, 288, 256, 32),
    (128, 576, 64, 64),
    # ImageNet ResNet-ish
    (32, 147, 12544, 64),
    (32, 576, 3136, 64),
    (32, 1152, 784, 128),
    (32, 2304, 196, 256),
    (32, 4608, 49, 512),
    # batch-1
    (1, 4608, 49, 512),
    (1, 576, 3136, 64),
    # big square-ish
    (8, 1024, 1024, 1024),
    # FC-like (L == 1)
    (64, 512, 1, 1000),
]


def make_inputs(N, K, L, O, device="cuda", seed=0, noisy=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-128, 128, (N, K, L), generator=g).float()
    w = torch.randint(-128, 128, (K, O), generator=g).float()
    if noisy:
        x += torch.randn(x.shape, generator=g) * 0.3
        w += torch.randn(w.shape, generator=g) * 0.3
    go = torch.randn((N, O, L), generator=g)
    # LRE slopes: smooth-ish per-index linear-fit coefficients
    dx = torch.randn(256, generator=g)
    dw = torch.randn(256, generator=g)
    return (t.to(device) for t in (go, x, w, dx, dw))


def qidx(t):
    return (t.round().clamp(-128, 127).long() + 128)


def truth(go, x, w, dx, dw, dtype=torch.float64):
    wp = dx[qidx(w)].to(dtype)                       # [K, O]
    xp = dw[qidx(x)].to(dtype)                       # [N, K, L]
    god = go.to(dtype)
    gx = torch.einsum("nol,ko->nkl", god, wp)
    gw = torch.einsum("nol,nkl->ko", god, xp)
    return gx.float(), gw.float()


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def check_correctness(opt_op, ref_op, cfg_op):
    print("== correctness (vs fp64 ground truth) ==")
    ok = True
    small = [(2, 7, 13, 5), (3, 33, 40, 17), (1, 64, 1, 32), (2, 100, 31, 96),
             (4, 65, 1, 7), (2, 128, 17, 256), (1, 5, 200, 3)]
    for shape in small:
        N, K, L, O = shape
        for noisy in (False, True):
            go, x, w, dx, dw = make_inputs(N, K, L, O, seed=sum(shape), noisy=noisy)
            gx_t, gw_t = truth(go, x, w, dx, dw)
            for name, op in [("opt", opt_op), ("ref", ref_op),
                             ("cfg1", lambda *a: cfg_op(*a, 1)),
                             ("cfg2", lambda *a: cfg_op(*a, 2))]:
                gx, gw = op(go, x, w, dx, dw)
                ex, ew = rel_err(gx, gx_t), rel_err(gw, gw_t)
                if ex > 1e-5 or ew > 1e-5:
                    print(f"  FAIL {name} N{N} K{K} L{L} O{O} noisy={noisy} "
                          f"relerr gx={ex:.2e} gw={ew:.2e}")
                    ok = False
    print("  small shapes:", "pass" if ok else "FAIL")

    # bench shapes: grad_x vs reference (fp32-level agreement); grad_w vs an
    # fp64 ground truth for BOTH kernels (accumulation order differs, so the
    # right question is who is closer to the true sum, not opt-vs-ref)
    for (N, K, L, O) in SHAPES:
        go, x, w, dx, dw = make_inputs(N, K, L, O, seed=K + O)
        gx_r, gw_r = ref_op(go, x, w, dx, dw)
        gx_o, gw_o = opt_op(go, x, w, dx, dw)
        ex = rel_err(gx_o, gx_r)
        xp64 = dw[qidx(x)].double()
        gw_t = torch.einsum("nol,nkl->ko", go.double(), xp64).float()
        ew_r, ew_o = rel_err(gw_r, gw_t), rel_err(gw_o, gw_t)
        del xp64, gw_t
        tag = "ok" if (ex < 1e-4 and ew_o <= max(1.5 * ew_r, 1e-6)) else "FAIL"
        if tag == "FAIL":
            ok = False
        print(f"  N{N:>4} K{K:>5} L{L:>6} O{O:>5}: {tag}  "
              f"gx_vs_ref={ex:.1e} gw_vs_truth ref={ew_r:.1e} opt={ew_o:.1e}")
    print("  =>", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def bench_fn(fn, *args, warmup=5, iters=30):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_bench(opt_op, ref_op, cfg_op, sweep=False, iters=30):
    print("\n== benchmark (end-to-end op call, ms) ==")
    hdr = f"{'shape (N,K,L,O)':>28} {'ref':>9} {'claude':>9} {'speedup':>8}"
    if sweep:
        hdr += "  cfg1(single) cfg2(batched)"
    print(hdr)
    for (N, K, L, O) in SHAPES:
        go, x, w, dx, dw = make_inputs(N, K, L, O, seed=K + O)
        flops = 4 * N * K * L * O  # 2 GEMMs
        it = max(5, min(iters, int(4e12 / max(flops, 1))))
        t_ref = bench_fn(ref_op, go, x, w, dx, dw, iters=it)
        t_opt = bench_fn(opt_op, go, x, w, dx, dw, iters=it)
        line = (f"{str((N,K,L,O)):>28} {t_ref:9.3f} {t_opt:9.3f} "
                f"{t_ref/t_opt:7.2f}x")
        if sweep:
            t1 = bench_fn(cfg_op, go, x, w, dx, dw, 1, iters=it)
            t2 = bench_fn(cfg_op, go, x, w, dx, dw, 2, iters=it)
            line += f"  {t1:11.3f} {t2:12.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("building extensions...")
    build()
    ops = torch.ops.approxtorch
    ref_op = ops.bgemm_lre_backward
    opt_op = ops.bgemm_lre_backward_claude
    cfg_op = ops.bgemm_lre_backward_claude_cfg

    if not args.skip_check:
        if not check_correctness(opt_op, ref_op, cfg_op):
            sys.exit(1)

    run_bench(opt_op, ref_op, cfg_op,
              sweep=args.sweep, iters=10 if args.quick else 30)


if __name__ == "__main__":
    main()
