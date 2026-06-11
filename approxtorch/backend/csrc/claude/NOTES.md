# claude/bgemm_float_claude.cu — optimization log

Target: LUT-based approximate-multiplication BGEMM
(`bgemm_fake_int8_forward_cuda` in `../cuda/bgemm_float_gpt.cu`).
Machine: 2x RTX 6000 Ada (sm_89, 142 SMs, 128KB unified L1/SM, ~96MB L2),
CUDA 12.8, PyTorch 2.9.1+cu128.

New ops (same signature as the reference):

- `approxtorch::bgemm_fake_int8_forward_cuda_claude(x, w, lut) -> y`
- `approxtorch::bgemm_fake_int8_forward_cuda_claude_cfg(x, w, lut, cfg) -> y`
  (tuning hook: cfg -1 = auto, 0..18 = force NFLAT/XMK tile cfg,
   100..118 = force SFLAT tile cfg; see dispatch_cfg for the table)

Run `python bench.py` (in this directory) for correctness + benchmarks,
`--sweep` to sweep tile configs.

## What the kernel is bound by

Every output needs K LUT gathers (`lut[(xi<<8)|wi]`, 4B each from a 256KB
table). Tile loads, stores and quantization are noise by comparison; the
whole game is gather efficiency:

1. **One LUT row per gather instruction.** A LUT row (fixed xi) is 1KB
   contiguous. If all 32 lanes of a warp share the same xi (row) and only
   wi varies, a gather touches <= 32 sectors of one row instead of 32
   scattered rows. So the warp must span the *column* dimension and the
   row value must be warp-uniform -> threadIdx.x spans columns, each warp
   has a single threadIdx.y.
2. **Row amortization ∝ block column width.** Per (row value, k) the row's
   sector footprint is paid once per block and amortized over BN columns.
   BN=512 -> ~2B of L2 traffic per useful lookup; BN=64 -> ~14B. Wider
   column tiles are nearly always better; the column dimension must
   therefore be the *large* one (see modes below).
3. **L1 working set ≈ BM KB per resident block** (BM distinct row values
   per k-step, 1KB row each; only 256 distinct rows exist in total).
   ~6 blocks/SM resident -> BM=16 keeps ~96KB hot and wins over BM=64
   even on big shapes (measured: cfg13/15 beat cfg11 broadly).
4. **int16 LUT image**: the table has only 256 rows; as float it is 256KB
   (2x L1), as int16 it is 128KB (≈ fits L1) and every gather pulls half
   the sectors. Approximate 8x8 multiplier tables are integer-valued, so
   an int16 copy is built per call and validated on device; the main
   kernel branches on a grid-uniform device flag (no host sync). The
   int16->float conversion is exact, so results stay bit-identical.
   Non-integer LUTs silently fall back to the float path.

## Structure

- prepass: `quantize_to_u8_kernel` turns x and w into uint8 LUT indices
  once (reference instead converted from float inside the hot loop and did
  a full `w.t().contiguous()` every call). Main kernel re-reads 1B/elem
  instead of 4B.
- `prepare_lut_kernel<TRANSPOSE>`: single tiny per-call LUT preprocessing
  launch (int16 image + validity flag, fused with the transpose for SFLAT).
- main kernel `bgemm_lut_u8_kernel<BM,BN,BK,TM,TN,MODE>`: 256 threads
  (32 x 8), register tile TM x TN per thread, smem tiles
  `srow[BK][BM]`, `scol[BK][BN]`, BK=32.
- three addressing modes put the warp along the largest dimension:
  - `NFLAT`: rows = N*L flattened, cols = O   (used when O is large)
  - `SFLAT`: rows = O, cols = N*L flattened, LUT transposed (small O);
    flattening (n,l) makes narrow late-conv L irrelevant — the column
    tile crosses image boundaries.
  - `XMK`:  L == 1 (im2col'd FC / GEMV): x viewed as row-major [N, K]
- split-K (grid.z): when gx*gy < 192 blocks, K is chunked to reach ~284
  blocks; partials land in a workspace, `reduce_splits_kernel` sums them
  in fixed order (deterministic; bit-exact for integer LUTs while sums
  stay < 2^24).

## Numerics

Accumulation is ascending-k float adds exactly like the reference, so
non-split-K results are bit-identical to `bgemm_fake_int8_forward_cuda`.
bench.py verifies vs both the reference kernel and a pure-torch ground
truth (including non-integer x/w rounding and non-integer LUTs).

## Measured (bench.py, end-to-end op time incl. all prepasses)

Final (v7, int16 LUT path), uniform-random int8 data — the worst case for
LUT locality; both kernels get faster on realistic gaussian activations,
where the relative speedup is ~3x:

| shape (N,K,L,O) | ref ms | claude ms | speedup |
|---|---|---|---|
| 128,27,1024,16 | 0.277 | 0.054 | 5.1x |
| 128,144,1024,16 | 1.518 | 0.225 | 6.8x |
| 128,288,256,32 | 1.478 | 0.162 | 9.1x |
| 128,576,64,64 | 1.487 | 0.170 | 8.7x |
| 32,147,12544,64 | 17.80 | 2.26-2.62 | 6.8-7.9x |
| 32,576,3136,64 | 17.77 | 2.14-2.43 | 7.3-8.3x |
| 32,1152,784,128 | 16.49 | 1.89-2.15 | 7.7-8.7x |
| 32,2304,196,256 | 16.35 | 1.71-1.75 | 9.4-9.5x |
| 32,4608,49,512 | 16.63 | 2.16-2.19 | 7.6-7.7x |
| 1,4608,49,512 | 0.655 | 0.118 | 5.6x |
| 1,576,3136,64 | 0.554 | 0.088 | 6.3x |
| 8,1024,1024,1024 | 38.18 | 4.54-5.48 | 7.0-8.4x |
| 64,512,1,1000 | 0.304 | 0.045 | 6.8x |

(ranges = run-to-run variance across bench runs, clocks/thermal)

## History / what was tried

- v1: u8 prepass + register tiling + warp-along-O. 1-4.2x, but small
  grids (bs1, deep K) and L==1 regressed; O<=32 barely moved.
- v2: split-K + XMK; cfg sweep -> big-O shapes want BN=256.
  Split-K threshold matters: helps <96 natural blocks, hurts ~100-256.
- v3: SWAP mode (warp along L, transposed LUT) -> small-O shapes 4-6.5x.
- v4: flatten (n,l) into one axis (NFLAT/SFLAT); narrow L stops
  mattering; grid.z freed for split-K.
- v5: BN=512 cfgs (15/16) win broadly in SFLAT; small-BM preference
  (L1 working set); mode rule NL >= 4*O for SFLAT (near the boundary
  NFLAT + split-K wins).
- v6: int16 LUT image with device-side validity flag.
- v7: with the int16 LUT mostly L1-resident, the wide-BN amortization
  configs stopped winning; re-sweep showed cfg0 (32,128,32,4,4) dominant
  in SFLAT/XMK -> retuned tables (pick_cfg_sflat assumes the int16 path;
  float-LUT fallback would prefer cfg13/15, ~15% off — acceptable since
  non-integer LUTs are rare). Fused LUT transpose + int16 prep into one
  launch. BK=64 cfgs (17/18) tried, never won.

Ideas not (yet) pursued: cp.async double buffering (gathers dominate,
tile loads are <5% of traffic); fp16 LUT (not exact); sorting/binning by
row value (restructuring cost outweighs).

---

# claude/bgemm_lre_backward_claude.cu — LRE backward

Target: `bgemm_lre_backward` in `../cuda/bgemm_lre_backward.cu`.
New ops (same schema + a tuning hook):

- `approxtorch::bgemm_lre_backward_claude(go, x, w, dx, dw) -> (gx, gw)`
- `approxtorch::bgemm_lre_backward_claude_cfg(..., cfg)`
  (cfg -1 auto, 1 = single-gemm grad_w, 2 = batched grad_w)

Run `python bench_backward.py [--sweep] [--quick]` here.

## Why this one is NOT a custom-kernel problem

Unlike the forward (irreducible 2-D `lut[(xi,wi)]` gather), the backward
factorizes: `DX[q(w[k,o])]` doesn't depend on x, so

    W' = DX∘q(w)  ([K,O], K*O cheap lookups)   gx[n] = W' @ go[n]
    X' = DW∘q(x)  ([N,K,L], one pass over x)   gw    = Σ_n X'[n] @ go[n]^T

i.e. tiny elementwise LUT prepasses + pure fp32 GEMMs -> cuBLAS. The
reference is a hand-rolled 16x16 one-output-per-thread smem GEMM (~1-2
TFLOPs) and additionally pays a full `go.permute(0,2,1).contiguous()`.

## Structure

- `lut_map_flat_kernel`: out[i] = slut[q(in[i])], smem LUT, float4 loop.
  Used for W' (always) and X' natural layout (N==1 path).
- `lut_map_kp_kernel`: X' written directly in [K, N*L] layout (read and
  write both l-coalesced; one 32-bit div per element).
- `transpose_ol_batched_kernel`: classic 32x32 smem-tiled batched
  transpose go [N,O,L] -> [N,L,O]; skipped when L==1 or O==1 (layouts
  coincide).
- grad_x: one `cublasSgemmStridedBatched`, strideB=0 broadcasts W',
  consumes go in its natural [N,O,L] layout (no permute at all).
  L==1 collapses to a single Sgemm ([N,O]@W'^T, avoids N gemv batches).
- grad_w: single Sgemm `X'[K,P] @ go_nlo[P,O]` (P=N*L). N==1 instead
  uses one batched call straight into grad_w (no transpose/workspace).
- TF32 only if `torch.backends.cuda.matmul.allow_tf32` (default off);
  handle math mode saved/restored around the calls.

## Heuristic (measured, not the traffic model)

Predicted batched+sum (workspace [N,K,O] + sum over N) would win for
K << L; sweep says otherwise — the single big GEMM won every N>1 shape
(one large GEMM amortizes better than N small ones + reduce pass), e.g.
(128,27,1024,16): single 0.068ms vs batched 0.121ms. So auto = single
unless N==1. The batched path is kept behind cfg=2.

## Numerics

LUT indexing identical (`__float2int_rn` + clamp). cuBLAS reduction
order differs from the reference's sequential loop, so not bit-exact —
but consistently *closer* to the fp64 ground truth (gw relerr ~3-7e-7
vs reference ~1-11e-6 across bench shapes). gx came out bit-identical
to the reference on nearly all shapes anyway.

## Measured (bench_backward.py, end-to-end op, ms)

| shape (N,K,L,O) | ref ms | claude ms | speedup |
|---|---|---|---|
| 128,27,1024,16 | 5.33 | 0.071 | 74.5x |
| 128,144,1024,16 | 6.58 | 0.368 | 17.9x |
| 128,288,256,32 | 1.41 | 0.113 | 12.5x |
| 128,576,64,64 | 0.55 | 0.088 | 6.3x |
| 32,147,12544,64 | 20.95 | 1.901 | 11.0x |
| 32,576,3136,64 | 6.83 | 1.320 | 5.2x |
| 32,1152,784,128 | 4.13 | 0.817 | 5.1x |
| 32,2304,196,256 | 3.51 | 0.634 | 5.5x |
| 32,4608,49,512 | 3.26 | 0.703 | 4.6x |
| 1,4608,49,512 | 0.26 | 0.053 | 5.0x |
| 1,576,3136,64 | 0.27 | 0.037 | 7.3x |
| 8,1024,1024,1024 | 7.95 | 1.194 | 6.7x |
| 64,512,1,1000 | 0.10 | 0.026 | 3.7x |

Sanity: (32,576,3136,64) cfg1 traffic estimate (X' build 462MB r+w,
transpose 51MB, gemm reads ~257MB, gx write 231MB) ≈ 1.2ms @ ~800GB/s —
measured 1.32ms, i.e. near memory-bound optimal; little headroom left
short of fusing the lookup into a custom GEMM.

Ideas not pursued: two streams to overlap the gx GEMM with the gw chain
(both near memory-bound, little to gain); custom fused lookup+GEMM
(loses to cuBLAS); TF32 by default (changes numerics).

## int8/uint8 inputs + forward `_save` hook (plan A, 2026-06-10)

The backward op also accepts x/w as int8 (idx = v+128) or uint8 (idx = v,
the forward op's LUT-index image); gradients stay float and are
bit-identical to the float path (indices identical). Both LUT-map
kernels are templated on input dtype (char4/uchar4 vectorized).

`bgemm_fake_int8_forward_cuda_claude_save(x, w, lut) -> (y, xq, wq)`
additionally returns the forward's internal u8 quantized images for free
(they were always computed; plain op = `std::get<0>`). The lre autograd
Function (`nn/bgemm.py`) saves xq/wq instead of fp32 x/w: saved-activation
memory 4x down (e.g. B32 C64 56^2 k3 layer: 270 -> 105 MB held after
forward, the rest is fakequant's scaled_x + misc), backward X'-map reads
1B/elem instead of 4 (op-level ~1.2x on map-bound shapes), no cast kernels
anywhere. A first attempt cast fp32->int8 in python; the cast cost
(~0.36ms on that layer) ate the backward win — exposing the forward's
existing u8 image is the zero-cost route.

## implicit-im2col backward (plan B, 2026-06-10)

`bgemm_lre_backward_claude_im2col(go, x[N,C,H,W], w, dx, dw, kh, kw, sh,
sw, ph, pw, dilh, dilw)`: x is the PRE-unfold quantized image
(f32/int8/uint8); `lut_map_kp_im2col_kernel` computes im2col gather
indices on the fly while building X' [K, N*L] (out-of-bounds == unfold
zero padding == LUT index 128). The unfolded tensor is never
materialized in backward; the image is kH*kW smaller and L2-resident, so
overlap reads are free. grad_x stays in im2col space [N,K,L] (autograd's
unfold backward folds it). Conv2d_int8_decoupled passes
`xq_pre = x.detach().to(torch.int8)` (one ~40us cast, only when grads
are needed) + conv geometry; `bgemm_int8_lre(..., xq_pre, geom)` falls
back to the u8-unfolded-image path when they are None.

Verified bit-identical to the unfold path for f32/i8/u8 inputs across
geometries incl. non-square kernels, stride 2, dilation 2, asymmetric
padding, 1x1 (kernel-level), and at module level for k3s1/k3s2/1x1s1/
1x1s2/5x5 (vs both u8-save and fp32-save: torch.equal).

Memory (B32 C64 56^2 k3): held-after-forward 270 (fp32) -> 105 (u8
unfolded) -> 55 MB (implicit), i.e. 4.9x vs original. Speed: X' build
344.5us == kp kernel 345.5us after moving n=p/L out of the per-element
path (grid.y spans n; GPU int division is ~20 emulated instructions and
cost ~+95us at first). Step time equal within noise (fwd equal, bwd
+80us on a machine with +-0.1-0.3ms clock drift; GPU kernel sums equal,
same launch count, no allocator churn).

The cfg hook note: with the refactor, cfg=1 for N==1 keeps the
transpose+single-gemm route via allow_direct_n1=false; auto N==1 uses
the direct batch-1 GEMM as before.

## fakequant_claude.cu — fused per-tensor fake-quant (plan C, 2026-06-10)

`fakequant_per_tensor_claude(x, scale, qmin, qmax) -> (q, mask)` and
`fakequant_per_tensor_backward_claude(go, mask, scale) -> gx`. The
python reference (nn/fakequant.py per-tensor Function) ran 3 elementwise
kernels forward + ~5 backward and saved the fp32 pre-round tensor just
to recompute the STE mask. Fused: one kernel each way, saving a 1-byte
bool mask (4x less). Bit-identical: same fp32 division, nearbyintf
(round-half-even == torch.round), NaN-preserving comparison clamp, same
1e-12 scale floor in fwd and bwd. Verified on half-values, clip
boundaries, scale=0, 4-bit ranges, odd lengths: torch.equal everywhere;
conv-level grads bit-eq. Kernel-seq speed: fwd 9.5x, bwd 11.2x.
nn/fakequant.py dispatches to the fused op for CUDA fp32 and keeps the
python path as CPU fallback. The per-channel (weight) quantizer is left
unfused on purpose — weights are tiny.

Cumulative held-after-forward on B32 C64 56^2 k3 (lre):
270MB (fp32 era) -> 105 (u8 unfold) -> 55 (implicit im2col) -> 37MB
(+ fused fakequant) = 6.65x; what remains is essentially the int8 image
(6.4MB) + bool mask (6.4MB) + the live output y itself.

## conv-restructure: u8 im2col forward + cuDNN backward (2026-06-11)

Conv2d_int8_decoupled (lre) no longer goes through F.unfold at all; the
autograd Function (nn/bgemm.py `_conv2d_int8_lre`) takes the fake-quantized
IMAGE and returns image-space gradients.

Forward: the fp32 unfold + the bgemm kernel's fp32->u8 prepass were a pure
bridge (9 B/elem of traffic to produce a 1-byte index). New op `im2col_u8`
(this file set) unfolds the int8 image straight to u8 LUT indices
(padding -> 128), and `bgemm_fake_int8_forward_cuda_claude_save` now accepts
a uint8 x and skips the x prepass. Bit-identical y (round-to-nearest of
exact integers == +128 cast). im2col_u8 launch notes: cap total blocks at
~12k and stride k over gridDim.z (1-byte writes make block scheduling the
bottleneck), uchar4 stores when L % 4 == 0 (0.83 -> 0.23 ms on B64 C64 56^2).

Backward (k != 1): grad_x = fold(W'^T @ go) and grad_w = go @ X'^T are
EXACTLY conv2d backward-data/-weight with the 1-D LUT maps applied to the
operands (elementwise map commutes with unfold), so both go to cuDNN
implicit-GEMM kernels: no X' materialization, no [N,K,L] grad, no fold.
Padding subtlety: unfold's zero padding maps to dw[128] != 0 while cuDNN
pads with literal zeros -> `lut_map_pad` bakes lut[128] into an explicit
border and the weight pass runs with padding=0; the data pass keeps the
original geometry (input passed as a zero-storage new_empty(1).expand, the
same trick torch.nn.grad uses). Two separate convolution_backward calls
beat one call + slicing the padded grad_input by ~27%. cudnn.allow_tf32 is
forced off inside the Function (torch defaults it ON for conv and would
silently degrade grads); benchmark=True while shapes repeat in training.
1x1/s1/p0 keeps the cuBLAS op (cuDNN measured 0.82x there).

Accuracy (test_conv_decoupled.py): y bit-identical to the old pipeline on
13 geometries (k7 stem, stride-2, dilation, non-square, batch-1, no-bias,
eval); grads vs fp64 truth 1e-7..4e-7 rel — same order as the old path.
Speed/memory (bench_module.py, interleaved): whole training step
1.20-2.18x on k=3 shapes (B64 C64 56^2: 1.50x; CIFAR B128 C16: 2.18x),
1x1 parity by design; forward+backward peak memory 1207 -> 277 MB on
B64 C64 56^2 (4.4x), 196 -> 45 MB on CIFAR.

## ste = identity-LUT lre (2026-06-11)

STE's einsum backward (grad_x = W^T·go, grad_w = go·X^T on quantized
values) is exactly the LRE backward with DX[i] = DW[i] = i - 128, so
Conv2d_int8_decoupled (ste) now routes through the same conv-level
Function (`bgemm.conv2d_int8_ste` -> `_conv2d_int8_lre` with an identity
LUT held as a persistent=False module buffer). The padding subtlety
vanishes: identity lut[128] = 0 matches cuDNN's zero padding. Activation
saving drops from the fp32 unfolded [N,K,L] (+ fp32 w) to the int8 image
+ u8 wq. Whole step 1.04-2.53x (B64 C64 56^2: 2.24x, peak 1537 -> 277 MB;
CIFAR: 2.53x, 250 -> 45 MB); unlike lre, 1x1 also gains (1.19x) because
the old einsum path was unoptimized there. Same test matrix: y
bit-identical, grads 1e-7 rel vs fp64. bqsg64 is the only remaining
unfold user in the module.
