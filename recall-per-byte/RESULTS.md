# Recall-per-byte: beating remax StackedSignBit on embedding compactness

> ## ⚠️ CORRECTION — ITQ was already tested and rejected (remax#46 / PR#47)
>
> This experiment's "ITQ beats remax" headline does **not** hold up. A more
> rigorous prior study — **remax #46 / PR #47, closed unmerged, 2026-06-18,
> SPECTER2 n=10k** — already tested learned ITQ rotations on the stacked ladder
> and found they **lose** to parameter-free centered SimHash at every rung, with
> the deficit **growing with k** (R@10 −0.006 to −0.037, k=1→8). Two facts gut my
> result:
>
> 1. **My apparent win is the known in-corpus overfit artifact.** I fit the ITQ
>    rotation on the same 600 docs I evaluated. The prior proved this exact thing
>    inflates ITQ: *transfer rotations learned on a different corpus BEAT
>    in-corpus ones*, because foreign rotations are less aligned on the eval data
>    and so behave more like the random rotations the ladder wants. My +0.04 at
>    k=1 on 600 docs is alignment overfit, not retrieval value.
> 2. **I never tested the stacked ladder (k>1)** — the config remax actually
>    ships, and exactly where ITQ's loss is largest. The mechanism: k rotations
>    each minimizing the same sign-quant MSE converge to correlated signatures,
>    eroding the ladder's 1/k variance reduction. **Decorrelation across stacks is
>    the asset, not per-projection optimality.**
>
> PQ is likewise not novel here: `remex` (TurboQuant codebook compression) already
> owns that axis, and my PQ number is flattered by training a 256-centroid
> codebook on only 600 docs. **Net: this experiment largely re-derived known /
> already-rejected territory.** The one genuinely open lead is the prior's own
> parting question — a *decorrelating joint multi-rotation objective* — see
> "Salvage" at the bottom. Original (now-corrected) writeup follows.


**Origin.** After iterating on model-weight quantization (int8→int4→int2), a
generative-thinking move (random stimulus: *"river"*) reframed the goal: a river
drains a wide catchment through a *variable-width* channel and settles its load
into *ordered layers* — i.e. compactness is about **information per stored bit
(the rotation) and bits-where-information-is**, not the model's weight precision.
remax spends a uniform 256 B/doc through a *random* rotation; both are leaving
compactness on the table.

This tests two directions that fired, against the remax baseline, at matched
bytes/doc — reusing the cached NFCorpus fp32 Jina embeddings (600 docs, 120 qrel
queries), no re-embedding. `sweep.py`.

## Encoders (matched bytes/doc B ∈ {16,32,48,64})

- **remax** — `StackedSignBitQuantizer` (the shipped lib): centered, dim=8B, k=1
  sign bits (its real rotation), B bytes.
- **simhash** — centered random-hyperplane sign bits, 8B bits (random rotation).
- **itq** — centered, PCA→8B dims, **ITQ learned rotation**, sign (direction A).
- **pq** — **Product Quantization**, M=B subquantizers × 8-bit codebooks (dir B).

Metric: recall@10/@100 vs topical gold (same harness as `bench_nfcorpus.py`).
Queries are held out of all training.

## Results

| bytes | remax @10/@100 | simhash @10/@100 | **itq @10/@100** | **pq @10/@100** |
|---|---|---|---|---|
| 16 | 0.177 / 0.453 | 0.140 / 0.436 | **0.221 / 0.498** | **0.255 / 0.554** |
| 32 | 0.191 / 0.445 | 0.213 / 0.473 | **0.246 / 0.500** | **0.246 / 0.547** |
| 48 | 0.210 / 0.499 | 0.198 / 0.479 | **0.238 / 0.511** | 0.249 / 0.546 |
| 64 | 0.227 / 0.487 | 0.233 / 0.505 | **0.239 / 0.516** | 0.245 / 0.536 |

**Reference — shipped remax (dim=256, k=8 = 256 B/doc):** R@10 0.208, R@100 0.477
(from `jina-int8-remax_kb/bench_nfcorpus.py`, same embeddings/qrels).

### Findings

1. **16× compaction at equal-or-better recall.** PQ at **16 B/doc** (R@100 0.554,
   R@10 0.255) and ITQ at **16 B/doc** (0.498 / 0.221) both *beat* shipped remax
   at **256 B/doc** (0.477 / 0.208). Same embeddings, same queries — apples to
   apples.
2. **Learned rotation > random, at every budget.** itq > simhash > remax on R@100
   across 16–64 B. The only change from remax→itq is replacing the random
   rotation with a corpus-learned ITQ rotation; it's a drop-in for the *same*
   1-bit-sign storage format.
3. **PQ wins recall-per-byte but with a caveat.** PQ is best, especially at the
   smallest budget — but its 256-centroid codebooks are trained on only 600 docs,
   which flatters it (centroids ≈ memorize docs). On a 100k-doc corpus PQ would
   be relatively coarser. **ITQ carries no such caveat and still beats remax** —
   so ITQ is the trustworthy headline; PQ is the optimistic ceiling.

## Caveats

- Single corpus (NFCorpus), 600-doc subsample, 120 queries — directional, not a
  leaderboard number. Absolute recall is low (hard dataset, aggressive compression).
- ITQ's PCA caps at ~min(N,D)=599 bits, so it tops out ~74 B/doc — fine for the
  compactness regime (low end), not a high-byte method.
- ITQ/PQ both train on the database (standard); queries are held out. ITQ's
  rotation is far lower-capacity than a codebook, so less overfit-prone.
- remax's random rotation isn't *wrong* — it's data-agnostic and portable (no
  fit step, the `.kb` carries only a seed). ITQ trades that for a learned rotation
  the `.kb` must store (a dim×dim matrix) — cheap to amortize over a corpus, but
  a real format change.

## Salvage — the one genuinely open lead

ITQ-as-drop-in is dead (see correction banner). But the *mechanism* that killed it
points somewhere new and untested. ITQ lost because its k stacked rotations all
minimize the **same** objective → correlate → defeat the ladder's 1/k variance
reduction. The prior (remax#46) closed with the explicit open question:

> *"If revisited, the open question is a DECORRELATING joint multi-rotation
> objective."*

That is the real lead: rotations that are **better-than-random per stack AND
mutually decorrelated across stacks** — keeping the diversity the ladder needs
while adding alignment each random rotation lacks. ITQ optimized the wrong thing
(per-projection MSE); a joint objective with a cross-stack decorrelation penalty
(or orthogonality between the k rotations' principal axes) attacks the actual
failure mode. **Must be validated the way #46 was: on SPECTER2 n=10k with proper
transfer testing** (learn on corpus A, eval on B) to defeat the in-corpus overfit
that flattered my run, and **on the stacked ladder, not k=1**, with an explicit
kill criterion (must beat random SimHash at k≥4 under transfer).

Still untested from the generative move: variable-rate per-doc allocation
(channel width follows load), code-space dedup (oxbow lakes). These don't touch
the rotation, so the #46 pathology doesn't pre-condemn them.

## Reproduce

```bash
# needs the cached NFCorpus fp32 embeddings from jina-int8-remax_kb/ (run its
# bench_nfcorpus.py once) + remax/remax_kb on path. sklearn for PQ.
python3 sweep.py
```
