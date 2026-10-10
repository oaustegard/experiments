# q4 head-to-head: official Optimum ONNX vs ours (NFCorpus)

Runs the [remax_kb#23](https://github.com/oaustegard/remax_kb/issues/23) benchmark
that claude.ai couldn't (egress blocks both weight hosts — `cas-bridge.xethub-eu.hf.co`
and the GitHub release asset). CCotw reaches both.

**Question:** we shipped `JinaQ4ONNXEmbedder` (remax_kb#14, ~170 MB int4-blockwise +
int8 embedding-table mop-up). The model authors *already* ship `onnx/model_q4.onnx`
(137.8 MB, HF Optimum, same MatMulNBits int4 family). Before any HF upload of ours:
does the official q4 match or beat ours on retrieval fidelity to fp32?

## Setup

- **Models:** fp32 `onnx/model.onnx` (849 MB) + official `onnx/model_q4.onnx` (138 MB)
  from `jinaai/jina-embeddings-v5-text-nano-retrieval`; ours `model.q4.onnx` (170 MB)
  from the `oaustegard/jina-v5-nano-mirror` release `v5-nano-8a7f00aa`.
- **Corpus:** BEIR NFCorpus → 2058 docs / 100 queries (judged-relevant set exceeds the
  1500-doc floor). 4 vCPU, onnxruntime CPU provider.
- **Runner:** `run_batched.py` — imports the repo's `bench/bench_q4_official_vs_ours.py`
  verbatim and reuses every metric; the only change is mini-batching the forward pass
  (the one-shot `encode(all_docs)` OOMs at ~26 GB on the attention-mask `Expand`).
  That same fix was upstreamed into the bench script in the remax_kb PR.

## Result

| model | nDCG@10 | ΔnDCG vs fp32 | per-doc cos | recall@10 vs fp32-kNN | Spearman ρ | MB | encode s |
|---|--:|--:|--:|--:|--:|--:|--:|
| fp32 | 0.4408 | 0.0000 | — | — | — | 849.3 | 815.6 |
| ours-q4 | 0.4250 | −0.0158 | 0.9743 | 0.8620 | 0.9764 | 169.7 | 844.4 |
| **official-q4** | **0.4291** | **−0.0118** | **0.9763** | **0.8700** | **0.9801** | **138.0** | 816.5 |

## Verdict: ours is dominated — do not upload

The official q4 wins **every axis at once**: smaller (138 vs 170 MB, −19%) *and*
strictly more faithful to fp32 on all four quality metrics (nDCG, cosine,
recall@10-vs-fp32kNN, ρ), same direction on each. No axis favors ours.

The escape hatch we hoped for — "ours holds where official degrades, because our int8
embedding-table mop-up was a Jina-specific fix Optimum's generic q4 may skip" — is
**refuted**: official edges ours on fidelity, so Optimum handles the EuroBERT embedding
`Gather` at least as well. Per the issue's rule (`official ≥ ours AND smaller → dominated`):
**do not upload ours; close the upload loop.**

Differences between the two q4s are small (ΔnDCG 0.004, Δcos 0.002) and individually
near-noise at n=100 queries, but they are *consistent across four independent metrics*
and compound with the 19% size gap — so the dominance call is robust even if any single
metric is a tie.

## Side findings

- **Docstring claim holds.** `JinaQ4ONNXEmbedder` claims "NFCorpus per-doc cosine 0.975
  to fp32"; this independent, larger subsample measures ours at **0.9743** (≈0.974). Not
  contradicted.
- **q4 is not faster on CPU.** All three encode in ~13–14 min — int4 MatMulNBits dequant
  offsets the smaller footprint. The q4 win is download size, not latency; the docstring's
  "~2x faster CPU decode" did not reproduce here.
- **Prior-art lesson reinforced** (cf. memory `prior-art-check`, 2026-06-28): the upstream
  repo already shipped a smaller, better q4. Checking siblings *before* building would have
  saved the JinaQ4ONNXEmbedder effort.

## Reproduce

```bash
pip install onnxruntime tokenizers numpy datasets
# fetch fp32+official+tokenizer from HF, ours-q4 from the GH release (see download note)
python run_batched.py \
  --fp32 models/onnx/model.onnx --ours-q4 models/model.q4.onnx \
  --official-q4 models/onnx/model_q4.onnx --tokenizer models/tokenizer.json \
  --n-docs 1500 --n-queries 100 --batch 64
```

Model weights are gitignored (regenerable). `results.txt` holds the raw run output.
