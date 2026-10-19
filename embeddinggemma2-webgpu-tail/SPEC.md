# EG2 fused tail on WebGPU — specification

The tail is everything after the transformer: pooling, normalization, centring,
rotation, 2-bit quantization, packing, and the asymmetric scan that searches the
packed corpus. It runs on the same `GPUDevice` as onnxruntime-web, so an
embedding goes from ORT's output buffer to a packed code without a readback.

The reference is remex 2.1.0, `Quantizer(d=768, bits=2, seed=42, mean=mu)`
(default rotation `"rht"`, applied here as its dense matrix `q.R`). Every
number this spec asks for has a remex counterpart, and the checker compares
against remex, not against itself.

## Constants

| name | value |
|---|---|
| `D` | 768 |
| `BITS` | 2 |
| `WORDS` | 48 — u32 words per packed vector (192 bytes) |
| `boundaries` | 3 f32, from `params.json` |
| `centroids` | 4 f32, from `params.json` |
| `kappa` | f32, `q._cosine_factor()` = 0.939556… |
| `R` | f32 `[D*D]` row-major: rotated_j = Σ_k R[j*D+k] · x_k (dense rotation only) |
| `rht` | remex's plan for R: `B` = 256, `rounds` = 2, `perms` and `signs` `[rounds][D]` (`remex.rotation.rht_plan`) |
| `Rmu` | f32 `[D]` = R·mu |
| `mu` | f32 `[D]` corpus mean (caller-declared) |

The kernels are written for `D = 768` and a workgroup of 256 invocations.
They must not require the `subgroups` feature or `shader-f16`.

## Math (per vector)

Input `x` (one row of `[B, D]`), optional normalize flag.

1. If normalize: `x = x / max(sqrt(Σ x²), 1e-8)`. (EG2's `sentence_embedding`
   is already unit; the flag also serves the pooled path.)
2. `t_j = (R x)_j − Rmu_j` — the rotated residual `R(x − mu)`. The default
   rotation applies R as remex's `RHTOperator.rotate_rows` does: for
   `r = rounds−1 … 0`, an unnormalized Walsh–Hadamard transform over each
   256-wide block (pairs `(i, i+h)`, `h = 1, 2, …, 128`), then
   `z[perm_r[i]] = y[i] · sign_r[i] / 16`. In float32 this is bit-identical to
   remex's operator. The `dense` rotation computes `Σ_k R[j,k] x_k` instead.
3. `len = sqrt(Σ t_j²)` — remex `norms` (= ‖x − mu‖).
4. `mdot = Σ Rmu_j t_j` — remex `mean_dots` (= mu·(x − mu)).
5. `u_j = t_j / max(len, 1e-8)`.
6. `code_j = (u_j > b0) + (u_j > b1) + (u_j > b2)` — strict `>`, which is
   `np.searchsorted(boundaries, u, side="left")`.
7. `vlen = sqrt(Σ c[code_j]²)`, `vmu = Σ c[code_j] · Rmu_j`.
8. `scale = len / (max(vlen, 1e-12) · kappa)` — remex `_score_norms`.
9. `bias = mdot − scale · vmu` — remex `_row_bias`.

Query side (`q`, same normalize flag):

- `qr_j = Σ_k R[j,k] q_k`, `offset = Σ_k q_k mu_k`.
- LUT `lut[j*4 + c] = qr_j · centroids[c]` (3072 f32 = 12 KB).

Score of doc `i` for query `q`:
`score = scale_i · Σ_j lut[j*4 + code_ij] + bias_i + offset` — remex
`search_adc` returns exactly this quantity.

## Packing

Codes are packed in remex's byte order (`remex.pack(codes, 2)`): within a byte,
code `4b+m` sits at bit shift `6 − 2m` (first code in the high bits). Bytes go
into little-endian u32 words, so in word `w`, code index `16w + 4·bb + m` sits at
bit shift `8·bb + 6 − 2m` (`bb` = byte within the word, 0..3). Reading the word
array as bytes on a little-endian host gives remex's packed byte stream.

## Corpus layout on the GPU (word-major)

A corpus has a fixed `capacity`. Its packed words live in one u32 buffer
`words[w * capacity + i]` (word `w` of doc `i`), so neighbouring scan threads
read neighbouring addresses. Per-doc floats live in `aux[i * 4 + {0,1,2,3}]` =
`{scale, bias, len, mdot}`.

`encode` writes rows `start .. start+count−1` of a corpus, so a corpus is filled
by several encode calls at increasing offsets.

## Kernels

1. **pool** — `[B, S, D]` hidden states + `[B, S]` u32 mask → `[B, D]` masked
   mean (sum of rows with mask≠0, divided by their count; count 0 → zeros).
   No normalization here; `encode` normalizes.
2. **encode** — one workgroup per tile of 4 vectors, held in the lanes of a
   `vec4`, so each element of R is read once per tile. Steps 1–9 above.
   Reductions are workgroup tree reductions in a fixed order; every lane does
   the same operations in the same order as a one-vector kernel, so the output
   is bit-identical to one. Workgroup memory is exactly the 16 KB default limit.
   Writes packed words and aux.
3. **prep_query** — one workgroup per query. Writes the LUT and `offset`.
4. **scan** — one invocation per (doc, query); the query's LUT loaded into
   workgroup memory. Writes `scores[q * capacity + i]` for `i < n`.
5. **topk** — one workgroup per (chunk of 1024 scores, query): a bitonic sort
   in workgroup memory, best `k` written out. Order: higher score first, lower
   index on ties. The host repeats passes over the candidates until one chunk
   remains, so only `nq · k` pairs are read back. `k ≤ 256`.
6. **colsum** — sums a `[B, D]` batch into a `[D]` f32 accumulator, one
   invocation per column. The host turns it into `mu` (divide by row count) and
   computes `Rmu` itself. This is how the demo declares a mean without reading
   embeddings back.

Batches larger than 65 535 workgroups are split by the host.

## JS API (`tail.js`, ES module, no dependencies)

```js
const tail = await RemexTail.create(device, params);
// params: { R, Rmu, mu: Float32Array, boundaries: number[3],
//           centroids: number[4], kappa: number }
tail.setMean(mu)                       // recomputes Rmu on the host, uploads both
const corpus = tail.allocCorpus(capacity);
tail.pool(hiddenBuf, maskBuf, B, S) -> GPUBuffer [B*D] f32
tail.encode(inputBuf, count, corpus, start, { normalize: true, inputOffsetRows: 0 })
tail.search(queryBuf, nq, corpus, { normalize: true }) -> GPUBuffer scores [nq*capacity]
await tail.readCorpus(corpus, n)  -> { packed: Uint8Array(n*192), scale, bias, len, mdot }
await tail.readScores(buf, nq, corpus, n) -> Float32Array [nq*n]
await tail.topk(queryBuf, nq, corpus, n, k) -> [{ idx: Int32Array(k), score: Float32Array(k) }] (GPU top-k, k <= 256)
await tail.topkHost(queryBuf, nq, corpus, n, k) -> same, from a full score readback and a host partial select
tail.colsum(inputBuf, count, accBuf)
```

`readCorpus` returns `packed` in remex byte order (doc-major, 192 bytes per
doc), transposing from the word-major GPU layout.

`device` is supplied by the caller — normally onnxruntime-web's own device, so
ORT's `gpu-buffer` outputs are valid inputs here.

## Verification

`oracle/make_fixtures.py` builds fixtures from remex; `oracle/check_tail.py`
compares an output directory against them:

- **codes**: equal to remex, except at coordinates whose float64 distance to the
  nearest boundary is below 1e-5 (reported as excused flips).
- **scale, bias, scores**: within 1e-4 relative (5e-6 absolute floor) of a
  float64 recomputation from the GPU's own codes; and within 1e-4 of remex
  on rows without an excused flip.
- **len, mdot**: within 1e-5 relative of remex.
- **top-10**: identical to remex's ranking, except where the scores at ranks
  10 and 11 sit within the score tolerance.
- **pool**: within 1e-5 of the float64 masked mean.
