# Correctness tests for the claude LUT GEMM ops
# (gemm_fake_int8_forward_cuda_claude / gemm_fake_uint8_forward_cuda_claude):
#
#   int8:  y[m,n] = sum_k lut[(A[m,k]+128)*256 + (B[k,n]+128)]
#   uint8: y[m,n] = sum_k lut[A[m,k]*256 + B[k,n]]
#
# torch.mm layout A [M,K] x B [K,N] -> y [M,N]. Every shape is fed to BOTH
# ops from the same u8-domain data (int8 gets values-128), so one fp64
# gather-sum oracle checks both AND their bit-equivalence at once. All
# comparisons are EXACT: LUT ranges and K keep fp32 partial sums below 2^24
# (or on the 0.5 grid below 2^23), so any tiling / orientation / split-K
# order must reproduce the oracle bit-for-bit.
#
#  A. oracle sweep over shapes hitting: both orientations (XMK warp-on-N /
#     swapped warp-on-M), GEMV edges M=1 / N=1 / K=1 / all-1, tile-exact and
#     +/-1 sizes, split-K in both orientations, tall-skinny and wide
#  B. forced-cfg sweep: every dispatch entry x both orientations
#  C. orientation grid.y-overflow forcing (M or N > 524280 with the wrong
#     forced orientation must silently flip and stay correct)
#  D. degenerate shapes: M=0 / N=0 return empty, K=0 returns exact zeros
#  E. input paths: uint8-dtype A/B (prepass skipped / fused), non-contiguous
#     A, B as a transposed weight view W.t() (the Linear case), fp32 clamp
#  F. LUT images: int16, uint16 (u8 product table), fp32 fallbacks
#     (non-integer / mixed-range), exact-product LUTs vs fp64 torch.mm
#  G. _save contract: aq [M,K], bq [N,K] is the TRANSPOSED image, u8 aliasing
#  H. invalid inputs raise cleanly (no crash): shape mismatch, bad lut size,
#     3-D A, CPU tensors
import torch
import approxtorch as at

torch.manual_seed(0)
dev = 'cuda'
nfail = 0


def report(ok, msg):
    global nfail
    if not ok:
        nfail += 1
    print(f'{"PASS" if ok else "FAIL"} {msg}')


i8 = torch.ops.approxtorch.gemm_fake_int8_forward_cuda_claude.default
u8 = torch.ops.approxtorch.gemm_fake_uint8_forward_cuda_claude.default
i8_cfg = torch.ops.approxtorch.gemm_fake_int8_forward_cuda_claude_cfg.default
u8_cfg = torch.ops.approxtorch.gemm_fake_uint8_forward_cuda_claude_cfg.default
u8_save = torch.ops.approxtorch.gemm_fake_uint8_forward_cuda_claude_save.default


# fp64 oracle: y[m,n] = sum_k lut2d[a_idx[m,k], b_idx[k,n]], chunked to
# bound the [mc, kc, nc] gather tensor
def ref_gemm(a_idx, b_idx, lut, mc=4096, kc=32, nc=65536):
    lut2d = lut.view(256, 256).double()
    M, K = a_idx.shape
    N = b_idx.shape[1]
    y = torch.zeros(M, N, dtype=torch.float64, device=dev)
    for m0 in range(0, M, mc):
        m1 = min(M, m0 + mc)
        for n0 in range(0, N, nc):
            n1 = min(N, n0 + nc)
            for k0 in range(0, K, kc):
                k1 = min(K, k0 + kc)
                vals = lut2d[a_idx[m0:m1, k0:k1].unsqueeze(2),
                             b_idx[k0:k1, n0:n1].unsqueeze(0)]
                y[m0:m1, n0:n1] += vals.sum(dim=1)
    return y


def make_inputs(M, K, N):
    # u8-domain values incl. both endpoints; int8 op gets (values - 128)
    au = torch.randint(0, 256, (M, K), device=dev).float()
    bu = torch.randint(0, 256, (K, N), device=dev).float()
    if au.numel():
        au.view(-1)[0] = 0.0
        au.view(-1)[-1] = 255.0
    if bu.numel():
        bu.view(-1)[0] = 0.0
        bu.view(-1)[-1] = 255.0
    return au, bu


# integer LUT small enough that |sum over K| < 2^24 for every tested K
int_lut = torch.randint(-64, 64, (65536,), device=dev).float()


def check_both(M, K, N, tag):
    au, bu = make_inputs(M, K, N)
    ref = ref_gemm(au.long(), bu.long(), int_lut)
    yu = u8(au, bu, int_lut)
    yi = i8(au - 128.0, bu - 128.0, int_lut)
    report(torch.equal(yu.double(), ref), f'oracle uint8  {tag}')
    report(torch.equal(yi, yu), f'int8 == uint8 {tag}')


# ------------------------------------------------------------------------ A
shapes = [
    (97,  103,  67),     # odd sizes, auto XMK (M < 2N)
    (67,  103,  33),     # auto swapped (M >= 2N)
    (1,   64,   129),    # M = 1 (GEMV row)
    (129, 64,   1),      # N = 1 (GEMV col)
    (1,   1,    1),      # minimal
    (5,   1,    7),      # K = 1
    (256, 128,  512),    # tile-exact, XMK
    (513, 128,  255),    # +/-1 off tiles, swapped
    (8,   8192, 16),     # split-K, XMK orientation
    (16,  8192, 8),      # split-K, swapped orientation
    (20000, 64, 16),     # tall-skinny -> swapped (warp along M)
    (16,  64,   20000),  # wide -> XMK (warp along N)
]
for (M, K, N) in shapes:
    check_both(M, K, N, f'M{M} K{K} N{N}')

# split-K determinism: identical results across runs
au, bu = make_inputs(8, 8192, 16)
report(torch.equal(u8(au, bu, int_lut), u8(au, bu, int_lut)),
       'split-K deterministic')

# ------------------------------------------------------------------------ B
au, bu = make_inputs(97, 103, 67)
ref = ref_gemm(au.long(), bu.long(), int_lut)
for cfg in [0, 1, 2, 3, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18]:
    yx = u8_cfg(au, bu, int_lut, cfg)          # forced XMK
    ys = u8_cfg(au, bu, int_lut, cfg + 100)    # forced swapped
    report(torch.equal(yx.double(), ref) and torch.equal(ys.double(), ref),
           f'forced cfg {cfg:3d} XMK + swapped')

# ------------------------------------------------------------------------ C
# M too large for XMK's grid.y -> forced XMK must flip to swapped
M, K, N = 600000, 16, 8
au, bu = make_inputs(M, K, N)
ref = ref_gemm(au.long(), bu.long(), int_lut)
report(torch.equal(u8_cfg(au, bu, int_lut, 0).double(), ref),
       'grid.y overflow: forced XMK -> swapped')
# N too large for swapped's grid.y -> forced swapped must flip to XMK
M, K, N = 8, 16, 600000
au, bu = make_inputs(M, K, N)
ref = ref_gemm(au.long(), bu.long(), int_lut)
report(torch.equal(u8_cfg(au, bu, int_lut, 100).double(), ref),
       'grid.y overflow: forced swapped -> XMK')

# ------------------------------------------------------------------------ D
au, bu = make_inputs(0, 8, 5)
y = u8(au, bu, int_lut)
report(y.shape == (0, 5), 'M=0 -> empty y, no crash')
au, bu = make_inputs(5, 8, 0)
y = u8(au, bu, int_lut)
report(y.shape == (5, 0), 'N=0 -> empty y, no crash')
au, bu = make_inputs(3, 0, 4)
y = u8(au, bu, int_lut)
report(y.shape == (3, 4) and torch.equal(y, torch.zeros_like(y)),
       'K=0 -> exact zeros')

# ------------------------------------------------------------------------ E
au, bu = make_inputs(97, 103, 67)
ref = ref_gemm(au.long(), bu.long(), int_lut)
y = u8(au, bu, int_lut)
report(torch.equal(u8(au.to(torch.uint8), bu, int_lut), y), 'u8-dtype A')
report(torch.equal(u8(au, bu.to(torch.uint8), int_lut), y), 'u8-dtype B')
report(torch.equal(u8(au.to(torch.uint8), bu.to(torch.uint8), int_lut), y),
       'u8-dtype A and B')

# non-contiguous A (strided rows), B as W.t() view (the Linear-layer case)
a_nc = torch.repeat_interleave(au, 2, dim=0)[::2]
W = bu.t().contiguous()            # weight [N, K]
b_view = W.t()                     # [K, N] transposed view, non-contiguous
assert not a_nc.is_contiguous() and not b_view.is_contiguous()
report(torch.equal(u8(a_nc, b_view, int_lut), y), 'non-contig A + B = W.t() view')

# fp32 out-of-range values clamp to the index range
a_oob = au.clone(); a_oob.view(-1)[1] = 300.0; a_oob.view(-1)[2] = -7.0
report(torch.equal(u8(a_oob, bu, int_lut), u8(a_oob.clamp(0, 255), bu, int_lut)),
       'fp32 A clamp to [0,255]')

# ------------------------------------------------------------------------ F
# uint16-range LUT (u8 product tables): L1-resident u16 image, both orients
u16_lut = torch.randint(0, 65536, (65536,), device=dev).float()
u16_lut[0], u16_lut[1], u16_lut[2] = 0.0, 65535.0, 32768.0
au, bu = make_inputs(64, 64, 48)   # 65535*64 < 2^24
ref = ref_gemm(au.long(), bu.long(), u16_lut)
report(torch.equal(u8(au, bu, u16_lut).double(), ref), 'uint16-range LUT')
report(torch.equal(u8_cfg(au, bu, u16_lut, 100).double(), ref),
       'uint16-range LUT, swapped (transposed u16 image)')

# non-integer and mixed-range LUTs -> fp32 fallback
half_lut = torch.randint(-128, 128, (65536,), device=dev).float() * 0.5
ref = ref_gemm(au.long(), bu.long(), half_lut)
report(torch.equal(u8(au, bu, half_lut).double(), ref), 'non-integer LUT (fp32)')
mix_lut = u16_lut.clone(); mix_lut[7] = -3.0
ref = ref_gemm(au.long(), bu.long(), mix_lut)
report(torch.equal(u8(au, bu, mix_lut).double(), ref), 'mixed-range LUT (fp32)')

# exact-product LUTs == fp64 torch.mm
aa = torch.arange(256, device=dev).float()
prod_i8 = ((aa.view(-1, 1) - 128) * (aa.view(1, -1) - 128)).reshape(-1)
ai = torch.randint(-128, 128, (33, 256), device=dev).float()   # 128*128*256 < 2^24
bi = torch.randint(-128, 128, (256, 41), device=dev).float()
y = i8(ai, bi, prod_i8)
report((y.double() - ai.double() @ bi.double()).abs().max().item() == 0.0,
       'int8 exact-product LUT == torch.mm')
prod_u8 = (aa.view(-1, 1) * aa.view(1, -1)).reshape(-1)
au, bu = make_inputs(33, 128, 41)                              # 255*255*128 < 2^24
y = u8(au, bu, prod_u8)
report((y.double() - au.double() @ bu.double()).abs().max().item() == 0.0,
       'uint8 exact-product LUT == torch.mm')

# ------------------------------------------------------------------------ G
au, bu = make_inputs(19, 32, 11)
y, aq, bq = u8_save(au, bu, int_lut)
report(torch.equal(aq, au.to(torch.uint8)), '_save: aq == u8 A values')
report(torch.equal(bq, bu.t().contiguous().to(torch.uint8)),
       '_save: bq == transposed u8 B image')
report(torch.equal(y, u8(au, bu, int_lut)), '_save: y matches plain op')
a8 = au.to(torch.uint8)
_, aq2, _ = u8_save(a8, bu, int_lut)
report(aq2.data_ptr() == a8.data_ptr(), '_save: u8-dtype A -> aq aliases input')

# ------------------------------------------------------------------------ H
def raises(fn, msg):
    try:
        fn()
        report(False, f'{msg} (no error raised)')
    except RuntimeError:
        report(True, msg)

au, bu = make_inputs(8, 16, 8)
raises(lambda: u8(au, torch.randn(17, 8, device=dev), int_lut),
       'raises: K mismatch')
raises(lambda: u8(au.unsqueeze(0), bu, int_lut), 'raises: 3-D A')
raises(lambda: u8(au, bu, int_lut[:100]), 'raises: bad lut size')
raises(lambda: u8(au.cpu(), bu.cpu(), int_lut.cpu()), 'raises: CPU tensors')

print(f'\n{"ALL PASS" if nfail == 0 else f"{nfail} FAILURES"}')
raise SystemExit(1 if nfail else 0)
