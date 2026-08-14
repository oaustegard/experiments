# BEIR SciFact encoded with jina-v5-nano q4 — the TTT-Embed × remex/remax matrices

**Status: done — artifact committed.** This directory is a *data product*, not
a finding. It exists so that the actual experiment
([issue #33](https://github.com/oaustegard/experiments/issues/33)'s "what
happens next") can run in numpy on claude.ai, where the encode itself cannot.

## Why this ran here

claude.ai measured this encoder at **<2 docs/s on one core** (13 s session load,
64 of 5,183 docs in 32 s) and reaps detached jobs after ~100 s, so a 45–85 min
corpus encode has no way to finish there. Query-side encode is fine in-session;
corpus-side is not. Encoding once here and committing the matrices removes the
problem permanently.

## Artifact

| File | Shape | Bytes |
|---|---|---|
| `data/Dm.npy` | (5183, 256) float32 | 5.3 MB |
| `data/Q.npy` | (300, 256) float32 | 0.3 MB |
| `data/meta.json` | `doc_ids`, `q_ids`, `qrels` (+ encoder/dataset provenance) | 0.1 MB |

`doc_ids` and `q_ids` are strings, aligned to row order. `qrels` maps a query id
to its list of relevant doc ids.

```python
import json, numpy as np
Dm = np.load("data/Dm.npy"); Q = np.load("data/Q.npy")
meta = json.load(open("data/meta.json"))
scores = Q @ Dm.T                      # both sides are already L2-normalized
```

## Encode settings

Pinned to the 2026-07-08 codec eval so its fidelity numbers carry over:

- encoder `oaustegard/jina-v5-nano-mirror` @ `v5-nano-8a7f00aa`, asset
  `model.q4.onnx` (169,736,452 B, retrieval adapter merged)
- `dim=256` Matryoshka truncation, `max_length=384`
- doc text = `title + ". " + text`; prefixes `"Document: "` / `"Query: "`
- last-token pool (`mask.sum(-1) - 1`), then L2-normalize, fp32

## Sanity check

**fp32 nDCG@10 of raw retrieval (`Q @ Dm.T`) against the qrels = 0.7067.**

Inside the 0.60–0.72 band issue #33 pre-registered for SciFact with a small
retrieval encoder — near its top, which is the comfortable side. Binary gains,
ideal DCG over `min(|rel|, 10)`, averaged over the 300 judged queries —
trec_eval's `ndcg_cut_10` convention.

`recheck.py` scores that metric against five hand-computed cases *before*
letting it say anything about the data, including the two an implementation
usually gets wrong (a relevant document outside k, and an ideal DCG that must
cap at k rather than at `|rel|`). Otherwise "nDCG = 0.7067" only means the
metric agrees with itself. It then re-derives the number from the committed
`.npy` files and diffs it against the figure written above, so this file and
`data/` cannot drift apart.

## Two deviations from the spec, both forced and both verified

### 1. The corpus is rebuilt from AllenAI, because every HF host is blocked here

Issue #33 says to pull `BeIR/scifact` from Hugging Face and to "retry until it
lands on GCP", because claude.ai allowlists `us.gcp.cdn.hf.co` but not
`us.aws.cdn.hf.co`. **That is not this container's policy.** Here *all* HF hosts
— `huggingface.co`, `hf.co`, `datasets-server.huggingface.co`, `cdn-lfs.hf.co`
and **both** CDN legs — answer `403 to CONNECT` at the egress proxy. Retrying is
explicitly the wrong move for a policy denial, so the dataset came from the
upstream BEIR itself wraps: the AllenAI SciFact release at
`scifact.s3-us-west-2.amazonaws.com` (reachable).

Two mappings in that rebuild are not guessable:

- **BEIR's *test* split is AllenAI's *dev* claims.** AllenAI's test claims ship
  unlabelled, which is why BEIR used dev.
- **Relevance is `cited_doc_ids`, not `evidence`.** `evidence` is the strictly
  smaller SUPPORT/CONTRADICT subset: 209 pairs over 188 queries, against BEIR's
  published 339 over 300.

`verify_shape()` asserts the reconstruction against BEIR's four published
counts before a single vector is computed, and all four match exactly:

```
ok  docs: 5183 (BEIR: 5183)
ok  claims (train+dev): 1109 (BEIR: 1109)
ok  test queries: 300 (BEIR: 300)
ok  judged test queries: 300 (BEIR: 300)
ok  test qrels pairs: 339 (BEIR: 339)
```

Hitting 339 rather than 340 is load-bearing: one dev claim cites the same
document twice, and matching BEIR's count is what says the dedup is right.

**What this check cannot see** (registered in `ANCHORS.md`): it pins the
*split*, not the *strings*. A wrong `title`/`abstract` join would pass it, and
would then be caught only by the nDCG band — which is far too loose to notice a
few points. A per-document diff against the real `BeIR/scifact` is the missing
check, and no reachable host here permits it. Anyone re-running this where HF
*is* allowlisted should do that diff and record the result.

### 2. The mirror's `embed_onnx.py` cannot load the q4 asset

The issue points at `scripts/embed_onnx.py` as the torch-free loader, but its
`materialize()` hardcodes the 847 MB fp32 `model.onnx` release asset — it will
neither fetch nor load the q4 asset this experiment is pinned to. Its pooling,
prefixing and normalization semantics are replicated verbatim in
`encode_scifact.py::Encoder._pool` instead, against the same `tokenizer.json`
and the same `pad_id=128001`.

## What broke

- **Checkpoint writer, caught by its own atomic rename.** `np.save(tmp, acc)`
  appends a second `.npy` when the path does not already end in one, so it
  wrote `.ckpt_docs.npy.part.npy` and the following `tmp.replace(ckpt)` raised
  `FileNotFoundError` — 40 s into a 20-minute run. Cost nothing because it
  failed loudly and immediately; had the rename been a copy-if-exists it would
  have silently produced no checkpoints and lost the run to the first reap.
  Fixed by writing through a file handle.
- Nothing else. The encode ran clean on the second attempt.

## Cost and wall-clock

| | |
|---|---|
| Encode | 19.8 min for 5,483 texts (4.6 texts/s overall), 4 vCPU / 15 GB |
| — documents | 5,183 at 4.4 docs/s (near the 384-token cap) |
| — queries | 300 at 44.3 q/s (claims are short) |
| ONNX session load | 1.5 s (claude.ai: 13 s) |
| Model download | 169.7 MB from the GitHub release |
| Corpus download | 3.1 MB from AllenAI S3 |
| API cost | none — no LLM calls |

**The win here is not speed per core.** At 4.4 docs/s on 4 vCPU this box does
~1.1 docs/s/core, against claude.ai's <2 docs/s on one — no better, plausibly
worse. Whole-machine throughput is only ~2.2× and the corpus encode still takes
20 minutes. What actually makes the job possible is that **20 minutes is
available at all**: claude.ai reaps detached jobs after ~100 s, which caps any
in-session encode at roughly 200 documents no matter how fast the core is.
Moving this off claude.ai bought a reaper-free process, not a faster one — and
that framing matters for the next encode, because throwing cores at it will not
help much.

The 8.7× session-load gap (1.5 s vs 13 s) points the same way: most of what
claude.ai loses on a short job is fixed overhead, not compute.

## Reproducing

```bash
python3 -m pip install --break-system-packages numpy onnxruntime tokenizers
python3 encode_scifact.py --fetch                     # model + corpus -> ~/.cache
python3 encode_scifact.py --check-batch-invariance    # encode -> data/
```

The run is resumable: it checkpoints every 400 texts and restarts from the last
checkpoint, because CCotw reaps idle background jobs.

## Determinism note — mini-batching is bit-exact

Encoding is mini-batched (`--batch`, default 16) because a one-shot batch makes
the attention-mask `Expand` allocate tens of GB and OOM; `q4-official-vs-ours`
hit exactly this at 1,500 docs and argued from first principles that per-row
output is unaffected. `--check-batch-invariance` measures it rather than
arguing: encoding the same 32 documents at batch 32 and at batch 4 gives
**max|Δ| = 0.000e+00** across all 256 dims. So batch size here is a pure
throughput knob and the artifact does not depend on it.

This does **not** extend to length-sorted bucketing, which reorders rows and
would need its own check. It was not used.

## Next (not this directory)

claude.ai picks up the `.npy` files and runs the experiment in numpy: learn the
TTT-Embed task vector under an oracle (qrels) teacher, sweep the document index
through remex 8/4/2/1-bit and remax k=8/4/2, and measure whether the delta
survives quantization — plus two label-free baselines (Rocchio, and the
mean(docs) − mean(queries) gap direction). Paper: arXiv 2608.12569; Muninn
memory `7461f178`.
