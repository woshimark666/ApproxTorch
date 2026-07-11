# Correctness tests for the uint8 BGEMM forward
# (op: bgemm_fake_uint8_forward_cuda_claude, uint8 x uint8 multiplier):
#
#   y[n,o,l] = sum_k lut[x_u8[n,k,l] * 256 + w_u8[o,k]]
#
# both operands are raw unsigned values in [0,255] (index = value, no offset).
#
# All comparisons are EXACT (torch.equal / max-abs == 0): LUT values and K are
# chosen so every fp32 partial sum is exactly representable (integer sums
# < 2^24, or 0.5-grid sums < 2^23), so any accumulation order — including
# split-K — must reproduce the double-precision oracle bit-for-bit.
#
#  A. independent oracle: chunked double-precision gather-sum over LUT indices,
#     across a shape sweep hitting XMK / NFLAT / SFLAT, split-K, K tails,
#     tile-boundary +/-1 sizes, N=O=L=1 edges
#  B. bit-equivalence with the int8 op: uint8(x, w) == int8(x - 128, w - 128)
#     (identical index streams -> identical kernels must agree bitwise)
#  C. forced-cfg sweep: every dispatch table entry x {NFLAT, SFLAT(+100)} on an
#     odd-sized shape, plus the forced-NFLAT -> SFLAT grid.y-overflow fallback
#  D. input-path variants: uint8-dtype x (prepass skipped, xq aliases input)
#     vs fp32 x, non-contiguous x / w, fp32 clamp of out-of-range values
#  E. LUT images: non-integer LUT (fp32 fallback), > int16-range LUT,
#     exact-product LUT lut[a][b] = a*b vs a plain einsum
#  F. _save returns: wq == raw u8 w values, xq == raw u8 x values / alias
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


def op_u8(x, w, lut):
    return torch.ops.approxtorch.bgemm_fake_uint8_forward_cuda_claude.default(x, w, lut)


def op_u8_cfg(x, w, lut, cfg):
    return torch.ops.approxtorch.bgemm_fake_uint8_forward_cuda_claude_cfg.default(x, w, lut, cfg)


def op_u8_save(x, w, lut):
    return torch.ops.approxtorch.bgemm_fake_uint8_forward_cuda_claude_save.default(x, w, lut)


def op_i8(x, w, lut):
    return torch.ops.approxtorch.bgemm_fake_int8_forward_cuda_claude.default(x, w, lut)


# double-precision oracle, chunked over K and L to bound memory:
# y[n,o,l] = sum_k lut2d[x_idx[n,k,l], w_idx[o,k]]
def ref_bgemm(x_idx, w_idx, lut, kc=16, lc=32768):
    lut2d = lut.view(256, 256).double()
    N, K, L = x_idx.shape
    O = w_idx.shape[0]
    y = torch.zeros(N, O, L, dtype=torch.float64, device=dev)
    for l0 in range(0, L, lc):
        l1 = min(L, l0 + lc)
        for k0 in range(0, K, kc):
            k1 = min(K, k0 + kc)
            vals = lut2d[x_idx[:, None, k0:k1, l0:l1], w_idx[None, :, k0:k1, None]]
            y[:, :, l0:l1] += vals.sum(dim=2)
    return y


def make_inputs(N, K, L, O):
    # full index ranges: x and w both in [0,255] incl. both ends
    x_u = torch.randint(0, 256, (N, K, L), device=dev).float()
    w_u = torch.randint(0, 256, (O, K), device=dev).float()
    x_u.view(-1)[0] = 0.0
    x_u.view(-1)[-1] = 255.0
    w_u.view(-1)[0] = 0.0
    w_u.view(-1)[-1] = 255.0
    return x_u, w_u


# integer LUT small enough that |sum over K| < 2^24 for every tested K
int_lut = torch.randint(-64, 64, (65536,), device=dev).float()


# ---------------------------------------------------------------- A + B + D
# shape sweep: (N, K, L, O) — comments give the mode/feature each one targets
shapes = [
    (7,   64,   1,    33),   # XMK
    (1,   32,   1,    1),    # XMK minimal (O=1)
    (128, 4096, 1,    8),    # XMK + split-K (few blocks, many k-tiles)
    (2,   49,   49,   512),  # NFLAT, large O branch (C >= 192)
    (3,   97,   53,   67),   # NFLAT, odd sizes, K tail (97 % 32 != 0)
    (1,   1,    13,   5),    # K = 1 (single k-tile, tail-only)
    (4,   288,  3136, 16),   # SFLAT (N*L >> O)
    (1,   4608, 8,    16),   # NFLAT + split-K, large-K tail
    (5,   33,   17,   129),  # off-tile-boundary R and C
    (2,   64,   640,  64),   # SFLAT threshold region (NL = 2O boundary area)
    (1,   16,   1,    1),    # everything minimal but K > 1
]

for (N, K, L, O) in shapes:
    x_u, w_u = make_inputs(N, K, L, O)
    ref = ref_bgemm(x_u.long(), w_u.long(), int_lut)

    # A: fp32 x path vs oracle
    y = op_u8(x_u, w_u, int_lut)
    report(torch.equal(y.double(), ref), f'oracle fp32-x   N{N} K{K} L{L} O{O}')

    # D: uint8-dtype x path (values are already LUT indices)
    y8 = op_u8(x_u.to(torch.uint8), w_u, int_lut)
    report(torch.equal(y8, y), f'u8-dtype x path N{N} K{K} L{L} O{O}')

    # B: bit-equivalence with the int8 op on the shifted signed image
    yi = op_i8(x_u - 128.0, w_u - 128.0, int_lut)
    report(torch.equal(yi, y), f'int8-op equiv   N{N} K{K} L{L} O{O}')

# D: non-contiguous inputs (strided slices; host contiguous()-normalizes)
x_u, w_u = make_inputs(3, 40, 30, 20)
x_nc = torch.repeat_interleave(x_u, 2, dim=2)[:, :, ::2]
w_nc = torch.repeat_interleave(w_u, 2, dim=1)[:, ::2]
assert not x_nc.is_contiguous() and not w_nc.is_contiguous()
report(torch.equal(op_u8(x_nc, w_nc, int_lut), op_u8(x_u, w_u, int_lut)),
       'non-contiguous x / w')

# D: fp32 prepass clamps out-of-range values to [0, 255]
x_u, w_u = make_inputs(2, 24, 11, 9)
x_oob = x_u.clone(); x_oob.view(-1)[1] = 300.0; x_oob.view(-1)[2] = -7.0
x_cl = x_oob.clamp(0, 255)
report(torch.equal(op_u8(x_oob, w_u, int_lut), op_u8(x_cl, w_u, int_lut)),
       'fp32 x clamp to [0,255]')

# ------------------------------------------------------------------------ C
# forced-cfg sweep on an odd shape: every dispatch entry, NFLAT and SFLAT
x_u, w_u = make_inputs(3, 97, 53, 67)
ref = ref_bgemm(x_u.long(), w_u.long(), int_lut)
for cfg in [0, 1, 2, 3, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18]:
    y = op_u8_cfg(x_u, w_u, int_lut, cfg)
    report(torch.equal(y.double(), ref), f'forced cfg {cfg:3d} (NFLAT)')
    y = op_u8_cfg(x_u, w_u, int_lut, cfg + 100)
    report(torch.equal(y.double(), ref), f'forced cfg {cfg + 100:3d} (SFLAT)')

# C: forced-NFLAT with huge N*L -> grid.y overflow -> auto SFLAT fallback
N, K, L, O = 2, 8, 270000, 8          # N*L = 540000 > 8 * 65535
x_u = torch.randint(0, 256, (N, K, L), device=dev).float()
w_u = torch.randint(0, 256, (O, K), device=dev).float()
ref = ref_bgemm(x_u.long(), w_u.long(), int_lut, kc=8)
y = op_u8_cfg(x_u, w_u, int_lut, 0)   # cfg 0 forces NFLAT; must fall back
report(torch.equal(y.double(), ref), 'grid.y-overflow NFLAT->SFLAT fallback')

# ------------------------------------------------------------------------ E
# non-integer LUT -> fp32-LUT fallback (0.5 grid keeps fp32 sums exact)
x_u, w_u = make_inputs(2, 64, 33, 21)
half_lut = torch.randint(-128, 128, (65536,), device=dev).float() * 0.5
y = op_u8(x_u, w_u, half_lut)
ref = ref_bgemm(x_u.long(), w_u.long(), half_lut)
report(torch.equal(y.double(), ref), 'non-integer LUT (fp32 fallback)')

# integer LUT beyond int16 range -> fp32-LUT fallback (K small: sums exact)
big_lut = torch.randint(-64, 64, (65536,), device=dev).float()
big_lut[123 * 256 + 45] = 40000.0
big_lut[7 * 256 + 200] = -40000.0
x_u, w_u = make_inputs(2, 8, 17, 13)
y = op_u8(x_u, w_u, big_lut)
ref = ref_bgemm(x_u.long(), w_u.long(), big_lut)
report(torch.equal(y.double(), ref), 'LUT beyond int16 (fp32 fallback)')

# exact-product LUT lut[a][b] = a * b: op == plain einsum
# (values reach 255*255 = 65025 > int16 range -> exercises the fp32-LUT path,
#  which is also the path real uint8 x uint8 multiplier tables will take)
aa = torch.arange(256, device=dev).float()
prod_lut = (aa.view(-1, 1) * aa.view(1, -1)).reshape(-1)
x_u, w_u = make_inputs(3, 96, 25, 31)      # 255*255*96 < 2^24: fp32-exact
y = op_u8(x_u, w_u, prod_lut)
ref = torch.einsum('nkl,ok->nol', x_u.double(), w_u.double())
report((y.double() - ref).abs().max().item() == 0.0,
       'exact-product LUT == einsum(x_u, w_u)')

# ------------------------------------------------------------------------ F
x_u, w_u = make_inputs(2, 32, 19, 11)
y, xq, wq = op_u8_save(x_u, w_u, int_lut)
report(torch.equal(xq, x_u.to(torch.uint8)), '_save: xq == raw u8 x values')
report(torch.equal(wq, w_u.to(torch.uint8)), '_save: wq == raw u8 w values')
report(torch.equal(y, op_u8(x_u, w_u, int_lut)), '_save: y matches plain op')
x8 = x_u.to(torch.uint8)
y2, xq2, _ = op_u8_save(x8, w_u, int_lut)
report(xq2.data_ptr() == x8.data_ptr(), '_save: u8-dtype x -> xq aliases input')
report(torch.equal(y2, y), '_save: u8-dtype x -> same y')

print(f'\n{"ALL PASS" if nfail == 0 else f"{nfail} FAILURES"}')
raise SystemExit(1 if nfail else 0)
