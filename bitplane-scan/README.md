# Bit-plane dispersion scan

Exact top-K inner-product search over int8 vectors that reads the corpus one bit plane at a time and drops candidates as soon as a provable bound rules them out. The repo has a numpy reference (`reference.py`), WGSL compute kernels, and a self-contained WebGPU page (`web/index.html`).

## Verdict

- **Exactness.** The scan returned the same top-K as brute-force int8 search in every run: 8 datasets × 16 queries in numpy, plus 5 test-vector queries and 4 page-generated 100k queries on WebGPU (SwiftShader).
- **Clustered data, K = 10.** The scan reads 0.42-0.45x the bytes of brute force. After the first pass (3 planes) 8-25% of candidates survive. After pass 2 about 0.1-0.3% remain.
- **K = 100.** Pruning is weaker: the scan reads 0.52-0.57x of brute force.
- **Isotropic Gaussian data.** The bounds are too loose to prune early. Pass 1 keeps 100% of candidates, and the scan reads 0.82-0.88x of brute force. Gaussian data sits near the worst case: the top-K scores lie only about 3.7σ above the median.
- **Floor.** Pass 1 reads 3/8 of the corpus no matter how well later passes prune, so 3 planes up front puts a floor of about 0.40x under the exact scan. With well-separated data nearly the whole remaining cost is that first pass. Moving that floor needs a cheaper first pass, such as sign plus one plane. After 2 planes, though, the bound half-width is still 0.7-0.9x the gap between the top-K and median scores, and a 2-plane first pass keeps 98-100% of candidates.
- **GPU speed.** This was only measured on SwiftShader, a software rasterizer, so there are no real-GPU timings. On SwiftShader the scan was slower than brute force: 1.5 s vs 1.0 s at 100k × 768. The popcount inner loop does 8 AND+popcount operations per plane word, while brute force uses `dot4I8Packed`. Whether reading fewer bytes beats doing more ALU work depends on the hardware. That is the open question for a real GPU.

## Algorithm

Each corpus vector `x_i` is stored as int8 with a per-vector float scale `s_i` (symmetric: `s_i = max|x_i| / 127`, components clamped to [-127, 127]). The query `q` is int8 too. The score is `s_i · (q · x_i)`. Ranking uses this score: it equals the dot product with the dequantized vector, scaled by the query's scale, which is constant per query.

**Bit planes: two's complement.** For a byte `x` with bits `b_0..b_7`, `x = Σ_{b<7} 2^b·b_b − 128·b_7`. Plane `b` of vector `i` is the D-bit mask of bit `b` over all components, packed into `D/32` u32 words. The planes are stored as `[plane][vector][word]`, so a pass that reads plane `b` touches one contiguous `N·D/8`-byte slab.

The query is split the same way into 8 planes `Q_c`. Then

```
q · plane_b(x) = Σ_c w_c · popcount(Q_c & X_b),   w_c = 2^c (c < 7), w_7 = −128
```

so every partial dot product is exact integer arithmetic built from `countOneBits` on u32 words.

Two's complement was chosen over sign + magnitude for three reasons:

1. **The unread remainder has a simple shape.** After reading the top `8−r` planes, the unread part is `e_r(x) = x & (2^r − 1)`, an unsigned value in `[0, 2^r − 1]` for every component, whatever the sign. With sign + magnitude the remainder's sign depends on the sign plane, so the bound has to carry per-component signs.
2. **The top planes form a fixed-point prefix.** The top `k` two's-complement planes hold `x` rounded toward −∞ to a multiple of `2^(8−k)`. The partial score is therefore a coarse quantization of the true score, and every pass refines it the same way.
3. **The query needs no special handling.** With sign + magnitude the query must be split into sign and magnitude as well (or use 2 popcounts per plane pair) to keep the inner loop integer-only. Two's complement uses 8 plain AND+popcounts per word with one negative weight.

The cost: in two's complement small negative values have high bits set, so the top planes of a near-zero negative component are "busy". This does not affect correctness or the bound below.

**Passes.** Pass `p` reads the next `k_p` planes (from the top down) for the surviving candidates only, adds them to the stored integer partial dot `q · hi_r(x)`, and computes lower and upper bounds (`r` = planes still unread):

```
q · x = q · hi_r(x) + q · e_r(x)
      = q · hi_r(x) + c_r · Σq + q · (e_r(x) − c_r·1),        c_r = (2^r − 1)/2
|q · (e_r(x) − c_r·1)| ≤ min( c_r · ‖q‖₁ ,  ‖q‖₂ · R_r(x) )
R_r(x) = ‖e_r(x) − c_r·1‖₂          (precomputed per vector, 7 floats)
```

The first term of the min is the interval bound, from `|e − c| ≤ c` per component. The second is Cauchy-Schwarz, using a per-vector norm of the centred remainder that is stored at index time (`7·4` bytes per vector, rounded up to the next float32). For D = 768 the Cauchy-Schwarz term is roughly 25% smaller than the interval term for typical vectors (about 255·‖q‖₂ vs 340·‖q‖₂ at r = 5). Centring at `c_r` halves the interval term and shrinks `R_r` compared with the uncentred remainder.

Scores are `s_i · (centre ± w)`. To make the comparison safe in float32, the half-width `w` is inflated by `×1.00001 + 1e-6·|centre| + 4`. `reference.py` checks that `lb ≤ exact ≤ ub` holds for every vector at every `r` on each dataset.

**Pruning rule.** θ is the K-th largest lower bound among the current candidates. A candidate survives if `ub ≥ θ`. This is safe: at least K candidates have exact scores ≥ θ, so any candidate with `ub < θ` cannot be in the top K. On the final pass (`r = 0`) `lb = ub = exact score`, so the survivors of `key ≥ θ` are exactly the top-K, plus ties. The top-K is then taken by (score desc, index asc), the same tie rule brute force uses.

**Plane budget.** If the scan stops after `B < 8` planes, the final ranking uses the bound midpoint `s·(q·hi + c_r·Σq)`, and the result is reported as recall@K against brute force.

**Bytes counted.** Every storage-buffer read in the GPU pipeline: plane words for the survivors, scale, `R_r`, carried index and partial dot, four 8-bit radix-select histogram passes over the lower-bound key, and the compaction read. Brute force is charged the int8 corpus, the scale and the same selection traffic. Writes and readback are not counted. The page and the reference use the same formula (`aux_read_bytes` / `auxReadBytes`).

## Measured results (numpy reference)

`python3 reference.py`: N = 100,000, D = 768, 16 queries per row, int8 per-vector scale. The schedule is `3,1,1,1,1,1` (pass 1 reads sign + top 2 planes, then one plane per pass) unless noted. Exact = top-K identical to brute force for all 16 queries. It held in every exact row. The script prints in/out counts and bytes for every pass of every schedule; `--json FILE` writes the same data as JSON.

Datasets:
- **gaussian iid**: i.i.d. N(0,1) components, queries from the same distribution.
- **clustered**: 1000 clusters on the unit sphere. Centres have a power-law spectrum (λ_j ∝ j⁻¹) in a random orthogonal basis plus a shared mean direction (a cosine of about 0.28 between random pairs, mimicking the "cone" of text embeddings). Within-cluster noise is half low-rank (32 directions) and half isotropic, at the stated spread. Queries are new draws from the mixture.
- **near-duplicate queries**: queries are corpus vectors plus small noise.

Real embeddings were not used: the Hugging Face CDN is blocked from the build container. The page accepts a float32 `.npy` file to run your own.

| dataset | K | survivors after each pass, % of N | exact bytes / brute (3,1,1,1,1,1) | exact bytes / brute (4,4) | recall@K, 3-plane budget | recall@K, 4-plane budget | pass-1 survivors, 6σ heuristic (not exact) | bytes / brute, 6σ heuristic |
|---|---|---|---|---|---|---|---|---|
| gaussian iid | 10 | 100.00 / 99.67 / 34.32 / 1.90 / 0.09 / 0.01 | 0.821 | 1.054 | 0.581 | 0.762 | 18.11 | 0.439 |
| clustered spread=0.35 | 10 | 7.93 / 0.13 / 0.10 / 0.10 / 0.06 / 0.01 | 0.419 | 0.526 | 0.506 | 0.756 | 0.10 | 0.405 |
| clustered spread=0.6 | 10 | 25.40 / 0.27 / 0.10 / 0.07 / 0.03 / 0.01 | 0.450 | 0.527 | 0.756 | 0.887 | 0.09 | 0.404 |
| clustered 0.35, near-dup queries | 10 | 12.05 / 0.18 / 0.10 / 0.09 / 0.06 / 0.01 | 0.426 | 0.526 | 0.644 | 0.800 | 0.10 | 0.405 |
| gaussian iid | 100 | 100.00 / 99.96 / 59.08 / 7.32 / 0.61 / 0.10 | 0.876 | 1.055 | 0.688 | 0.854 | 40.00 | 0.486 |
| clustered spread=0.35 | 100 | 51.08 / 12.26 / 1.62 / 0.32 / 0.13 / 0.10 | 0.520 | 0.590 | 0.968 | 0.986 | 1.02 | 0.407 |
| clustered spread=0.6 | 100 | 76.27 / 13.64 / 1.54 / 0.27 / 0.12 / 0.10 | 0.566 | 0.598 | 0.979 | 0.991 | 0.97 | 0.407 |
| clustered 0.35, near-dup queries | 100 | 66.19 / 19.35 / 2.85 / 0.44 / 0.13 / 0.10 | 0.561 | 0.628 | 0.975 | 0.988 | 1.75 | 0.408 |

Brute force reads 79.2 MB at this size (`N·(D + 24)` bytes).

**Bound tightness.** The table below gives the median bound half-width divided by the gap between the K-th best exact score and the median exact score (first query, K = 10). Pruning starts once this ratio drops below about 1.

| planes read | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| gaussian iid | 7.69 | 3.50 | 1.75 | 0.87 | 0.43 | 0.21 | 0.08 |
| clustered 0.35 | 1.56 | 0.71 | 0.35 | 0.18 | 0.09 | 0.04 | 0.02 |
| clustered 0.6 | 1.96 | 0.89 | 0.44 | 0.22 | 0.11 | 0.05 | 0.02 |

The half-width halves with every plane, as expected since `R_r ≈ 2^r·√(D/12)`. On clustered data the bound is useful from 3 planes on. On Gaussian data it only becomes useful at 5 planes.

**Other schedules.**
- `2,1,1,1,1,1,1` (sign + 1 plane first) prunes almost nothing in pass 1 (98-100% survive) and costs 4-7% more bytes than `3,1,…`.
- `4,4` is simple but reads more (0.53x vs 0.42x on clustered data).
- The "6σ heuristic" row replaces the provable half-width with `6·‖q‖₂·R_r/√D`, the expected size of the remainder dot product if remainders were independent of q. It found the exact top-K in all runs here, but **it is not a guarantee**. It shows the most a statistical bound could save on top of the provable one: 0.40-0.49x vs 0.42-0.88x.

## Files

| path | what |
|---|---|
| `reference.py` | numpy reference: quantize, pack planes, residual norms, bounds (float32, mirrored by the kernels), scan, brute force, experiment driver, test-vector export |
| `web/index.html` | self-contained page: inline WGSL, a JS port of the reference, the WebGPU driver and the UI (no build step) |
| `tools/check_gpu.py` | runs the page in headless Chromium (SwiftShader WebGPU) against vectors from `reference.py --export` |

### Kernels (all in `web/index.html`, `<script type="text/wgsl">` blocks)

| kernel | job |
|---|---|
| `score` | one thread per surviving candidate. Reads the next `k` planes, accumulates `Σ_c w_c·popcount(Q_c & X_b)` into the carried integer partial dot, writes `(fkey(lb), fkey(ub))` (or the exact or midpoint key on the final pass). `fkey` maps f32 to u32 in order. |
| `brute` | full int8 dot product with `dot4I8Packed` when the adapter has `packed_4x8_integer_dot_product`, otherwise a scalar fallback. Writes the same key format. |
| `hist` + `scan` | exact radix select of the K-th largest lower-bound key: 4 rounds of an 8-bit histogram (workgroup-local atomics, then a global merge) followed by a single-thread bucket choice. |
| `compact` | atomic-append compaction of candidates with `ub key ≥ θ` into a ping-pong buffer of (index, partial dot). On the final pass it appends (index, score key). |
| `indirect` | writes the next pass's dispatch size from the survivor count, so all passes run in one submit with no readback in between. |

The final ≥K survivors (K plus ties) are read back and sorted on the host by (score desc, index asc).

## What was verified, and where

Verified:
- **numpy**: the plane decomposition equals the direct int8 dot product. `lb ≤ exact ≤ ub` for every vector and every `r`. The exact scan equals brute-force top-K for all 8 datasets × 16 queries.
- **JS port vs numpy**: on 5 queries from `reference.py --export` (N = 4000, D = 256, K = 10, clustered), the JS CPU scan, the JS brute force and the per-pass survivor counts are identical to numpy.
- **WGSL**: all kernels pass `naga` (naga-cli, cargo), including both `dot4` variants, and compile without errors in Dawn (Chromium 1243).
- **WebGPU execution (SwiftShader only)**: through `tools/check_gpu.py`, GPU bit-plane top-K, GPU brute-force top-K and GPU per-pass survivor counts equal numpy on all 5 test-vector queries. This was checked with native `dot4I8Packed` and again with the scalar fallback. On a 100k × 768 corpus generated in the page (clustered, 4 queries), GPU scan top-K = GPU brute force = CPU brute force, and the GPU survivor counts match the JS reference pass by pass.
- **No-WebGPU path**: with `navigator.gpu` removed, the page shows the "WebGPU unavailable" panel and still runs the CPU reference.

Not tested:
- Any hardware GPU. All GPU numbers come from SwiftShader on a 2-core container and say nothing about real-GPU speed. SwiftShader timings at 100k × 768, clustered, K = 10: brute force about 0.97 s, bit-plane scan about 1.5 s. Pass 1 took 1.36 s of that.
- Firefox and Safari WebGPU.
- Very large N, where a buffer would exceed `maxStorageBufferBindingSize`. The page refuses with a message when `N·D` exceeds the limit.
- Real embedding corpora. You can run one through the page's `.npy` loader.

## Running the page

Open `web/index.html` in a WebGPU browser (Chrome or Edge 113+, Safari 26+, or Firefox with WebGPU enabled). It needs no server and no build step. On load it generates a 20k × 768 clustered corpus and runs it once. Use "Generate corpus" for other sizes and distributions, change "Planes per pass" and "Plane budget", and press Run. The result chips show whether GPU top-K matches brute force. The table shows survivors, bytes read and GPU time per pass. GPU times come from `timestamp-query` when the adapter supports it. To use your own embeddings, load a C-order float32 `.npy` of shape [N, D]. D is zero-padded to a multiple of 32, and the last rows are held out as queries.

The page's generator uses its own RNG (mulberry32), so its corpora are drawn from the same distribution as `reference.py`'s but do not match it number for number. The exact comparison against numpy goes through the exported test vectors.

Reproduce the reference and the GPU check:

```
python3 reference.py                       # full tables (about 10 min on 2 cores); --quick for N=20k
python3 reference.py --export /tmp/tv      # test vectors
python3 tools/check_gpu.py web/index.html /tmp/tv   # needs playwright + a Chromium build with SwiftShader
```
