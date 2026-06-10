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
