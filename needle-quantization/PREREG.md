# Pre-registration — what does Needle 2's quantization cost, and is there room either way?

Written before any arm was exported or scored.

## The premise, which is different from the pruning question

`needle-layer-pruning/` found essentially no slack in depth. Quantization is not
the same question, because **Needle 2 is already aggressively quantized**: the
checkpoint carries `weight_bits = "embedding=4,mhc=4,default=2"`, so the bulk of
the model ships at **2 bits** with two tensor families protected at 4. The
shipped 13.74 MB blob is that spec.

So there are two questions, not one:

1. **Up** — what does the 2-bit default cost? `default=4` is the ceiling the
   exporter allows (`CQ_BITS = (2, 3, 4)`), so this is measurable, not
   hypothetical.
2. **Down** — the exporter also supports **ternary** (`TERNARY_BITS = 1.58`,
   codebook `{-1.224, 0, +1.224}`). Is anything below 2 bits usable?

And one ablation of Cactus's own choice: **is protecting `embedding` and `mhc`
at 4 bits load-bearing**, or is it caution?

## What is held fixed

The unpruned 27-layer checkpoint, `needle-bsky/evalset.jsonl` (62 queries, 54
routable, 8 off-topic), the `tuned-min` schema arm, and `needle-bsky`'s scoring
code imported unchanged — the same configuration measured at **0.611** by
`needle-bsky`, `needle-tool-naming` and `needle-layer-pruning` alike. Scoring
runs through `needle-layer-pruning/eval_pruned.py` so the numbers are directly
comparable to that sweep.

Confidence is `None` on every `weights=` path, so the gate is out of scope here
exactly as it was for pruning, and the metric is routing accuracy.

## Arms

| arm | `weight_bits` spec | what it isolates |
|---|---|---|
| `shipped` | `embedding=4,mhc=4,default=2` | control — the shipped model |
| `all4` | `default=4` | the ceiling; what 2-bit costs |
| `all3` | `default=3` | |
| `all2` | `default=2` | is the 4-bit protection load-bearing? |
| `prot3` | `embedding=4,mhc=4,default=3` | protection kept, bulk raised |
| `prot-tern` | `embedding=4,mhc=4,default=1.58` | protection kept, bulk ternary |
| `all-tern` | `default=1.58` | the floor |
| `emb2` | `embedding=2,mhc=4,default=2` | embedding protection alone |
| `mhc2` | `embedding=4,mhc=2,default=2` | mhc protection alone |
| `engram-tern` | shipped + `engram0.tables=1.58,engram1.tables=1.58` | the 8.39M of n-gram tables `needle-layer-pruning` found cheap to destroy |

Blob size is recorded for every arm; it is half the point.

## Predictions

| # | prediction |
|---|---|
| P1 | `shipped` reproduces **0.611**, as the pruning control did |
| P2 | `all4` ≥ `shipped`, by **0.00 to 0.06** — some headroom, not a lot, at ~1.65x the bytes |
| P3 | `all2` < `shipped` by **≥ 0.05** — the 4-bit protection is load-bearing, not caution |
| P4 | `all-tern` collapses below **0.20** |
| P5 | `prot-tern` lands between, **0.30–0.50** |
| P6 | **`emb2` hurts more than `mhc2`** — the embedding is 4.19M parameters *and* is weight-tied to the output head (`logits = x @ embedding.T`), so quantizing it degrades the input representation and the logits at once |
| P7 | `engram-tern` costs **< 0.05** — the pruning sweep showed destroying a whole Engram table costs less than four well-placed layers, so ternarizing both should be close to free |

**The interesting outcome is P2 failing upward** — if `all4` is worth more than
6 points, the shipped CQ2 spec is trading real accuracy for bytes and a
developer with 23 MB to spare should know.

## Reported regardless

Bytes per arm, and bytes-per-accuracy-point against the control, since the whole
argument for a 45M model is that it fits somewhere a bigger one does not.
