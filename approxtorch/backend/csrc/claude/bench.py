#!/usr/bin/env python
"""Correctness + performance harness for the claude-optimized LUT BGEMM.

JIT-compiles both the reference kernel (cuda/bgemm_float_gpt.cu) and the
optimized kernel (claude/bgemm_float_claude.cu) and compares them across a
suite of shapes resembling im2col'd conv layers.

Usage:
  python bench.py                 # correctness + benchmark with auto cfg
  python bench.py --sweep         # additionally sweep all cfg ids per shape
  python bench.py --quick         # fewer iters
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
    ref = load(
        name="gpt_bgemm_ref",
        sources=[os.path.join(CUDA_DIR, "bgemm_float_gpt.cu")],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    opt = load(
        name="claude_bgemm_opt",
        sources=[os.path.join(HERE, "bgemm_float_claude.cu")],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    return ref, opt


# (N, K, L, O) — x:[N,K,L], w:[O,K]
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
    # batch-1 inference
    (1, 4608, 49, 512),
    (1, 576, 3136, 64),
    # big square-ish
    (8, 1024, 1024, 1024),
    # FC-like (L == 1)
    (64, 512, 1, 1000),
]


def make_inputs(N, K, L, O, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-128, 128, (N, K, L), generator=g).float().to(device)
    w = torch.randint(-128, 128, (O, K), generator=g).float().to(device)
    lut = (torch.randn(256 * 256, generator=g) * 100).round().float().to(device)
    return x, w, lut


def truth_small(x, w, lut):
    xi = (x.round().clamp(-128, 127).long() + 128)          # [N,K,L]
    wi = (w.round().clamp(-128, 127).long() + 128)          # [O,K]
    idx = xi.unsqueeze(1) * 256 + wi.unsqueeze(0).unsqueeze(-1)  # [N,O,K,L]
    return lut[idx].sum(dim=2)


def check_correctness(opt_op, ref_op):
    print("== correctness ==")
    ok = True
    # exact ground truth on small shapes (including non-integer inputs)
    for (N, K, L, O) in [(2, 7, 13, 5), (3, 33, 40, 17), (1, 64, 1, 32), (2, 100, 31, 96)]:
        x, w, lut = make_inputs(N, K, L, O, seed=N * 1000 + K)
        yt = truth_small(x, w, lut)
        yo = opt_op(x, w, lut)
        if not torch.allclose(yt, yo, atol=1e-3, rtol=1e-5):
            print(f"  FAIL vs truth  N{N} K{K} L{L} O{O}  maxdiff={(yt-yo).abs().max().item()}")
            ok = False
        # non-integer floats (test round-to-nearest-even path)
        x2 = x + torch.randn_like(x) * 0.3
        w2 = w + torch.randn_like(w) * 0.3
        yt2 = truth_small(x2, w2, lut)
        yo2 = opt_op(x2, w2, lut)
        if not torch.allclose(yt2, yo2, atol=1e-3, rtol=1e-5):
            print(f"  FAIL vs truth (noisy)  N{N} K{K} L{L} O{O}")
            ok = False
    # non-integer LUT (forces the float-LUT fallback path)
    for (N, K, L, O) in [(3, 33, 40, 17), (4, 200, 300, 96), (2, 128, 17, 256)]:
        x, w, lut = make_inputs(N, K, L, O, seed=7)
        lut = lut + torch.rand_like(lut) * 0.9
        yt = truth_small(x, w, lut)
        yo = opt_op(x, w, lut)
        if not torch.allclose(yt, yo, atol=1e-2, rtol=1e-5):
            print(f"  FAIL float-LUT fallback N{N} K{K} L{L} O{O} "
                  f"maxdiff={(yt-yo).abs().max().item()}")
            ok = False
    # contiguous views with nonzero storage_offset (misaligned data_ptr) must
    # take the scalar quantize path and match the float4 path bit-for-bit
    def off(t):
        base = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
        base[1:] = t.reshape(-1)
        return base[1:].view(t.shape)

    x, w, lut = make_inputs(3, 33, 40, 17, seed=5)
    if not torch.equal(opt_op(x, w, lut), opt_op(off(x), off(w), off(lut))):
        print("  FAIL offset-view inputs")
        ok = False
    print("  offset-view inputs:", "pass" if ok else "FAIL")
    # vs reference kernel on all bench shapes (expect bit-identical)
    for (N, K, L, O) in SHAPES:
        x, w, lut = make_inputs(N, K, L, O, seed=K + O)
        yr = ref_op(x, w, lut)
        yo = opt_op(x, w, lut)
        if torch.equal(yr, yo):
            tag = "bitexact"
        elif torch.allclose(yr, yo, atol=1e-2, rtol=1e-5):
            tag = f"close (maxdiff={(yr-yo).abs().max().item():.2e})"
        else:
            tag = f"FAIL maxdiff={(yr-yo).abs().max().item():.3e}"
            ok = False
        print(f"  N{N:>4} K{K:>5} L{L:>6} O{O:>5}: {tag}")
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
    hdr = f"{'shape (N,K,L,O)':>28} {'ref':>9} {'claude':>9} {'speedup':>8} {'GLU/s':>8}"
    if sweep:
        hdr += "  best-cfg"
    print(hdr)
    rows = []
    for (N, K, L, O) in SHAPES:
        x, w, lut = make_inputs(N, K, L, O, seed=K + O)
        lookups = N * K * L * O
        it = max(5, min(iters, int(2e12 / max(lookups, 1))))
        t_ref = bench_fn(ref_op, x, w, lut, iters=it)
        t_opt = bench_fn(opt_op, x, w, lut, iters=it)
        glu = lookups / (t_opt * 1e-3) / 1e9
        line = (f"{str((N,K,L,O)):>28} {t_ref:9.3f} {t_opt:9.3f} "
                f"{t_ref/t_opt:7.2f}x {glu:8.1f}")
        best = None
        if sweep:
            results = []
            for cfg in list(range(17)) + list(range(100, 117)):
                try:
                    t = bench_fn(lambda: cfg_op(x, w, lut, cfg), iters=max(5, it // 2))
                    results.append((t, cfg))
                except RuntimeError as e:
                    pass
            results.sort()
            best = results[0]
            line += f"  cfg{best[1]}={best[0]:.3f}ms"
            top = ", ".join(f"c{c}:{t:.3f}" for t, c in results[:4])
            line += f"  [{top}]"
        print(line)
        rows.append((N, K, L, O, t_ref, t_opt, best))
    return rows


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
    ref_op = ops.bgemm_fake_int8_forward_cuda
    opt_op = ops.bgemm_fake_int8_forward_cuda_claude
    cfg_op = ops.bgemm_fake_int8_forward_cuda_claude_cfg

    if not args.skip_check:
        if not check_correctness(opt_op, ref_op):
            sys.exit(1)

    run_bench(opt_op, ref_op, cfg_op,
              sweep=args.sweep, iters=10 if args.quick else 30)


if __name__ == "__main__":
    main()
