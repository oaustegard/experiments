# Error log — ttt-embed-quantized

Every error found in this run, how it was caught, and which direction it pushed
the conclusion. Read the **direction** column first: an error that makes a
result look *stronger* or a check *more permissive* is the dangerous kind,
because it never announces itself downstream.

This is a short log because this is a short experiment — an encode with a single
scalar sanity check, not a multi-arm study. That is itself worth stating: a low
count here is a claim about scope, not about care.

---

| # | Error | Caught by | Direction | Fixed |
|---|---|---|---|---|
| 1 | Assumed length-sorted batching gives "roughly 2x" and nearly wrote that into `RESULTS.md` and `METHODS.md` as a portable number, on intuition alone ("halves the padding") | Deciding to measure before publishing; a 240-doc A/B gave **1.37x** | **Overstated a portable claim** — would have put an unearned number in the ledger other experiments budget against | Yes — measured value reported, with the gap to intuition called out |
| 2 | First `repo-index/ask.py` call died `ModuleNotFoundError: numpy`, then `ModuleNotFoundError: remex` | The mandated pre-flight itself | Neutral — but this is the *exact* failure `METHODS.md` records under Environment gotchas, where it was once misdiagnosed as "`xr` is unavailable in this container" | Yes — `pip install --break-system-packages numpy onnxruntime tokenizers remex`, as the ledger prescribes. The entry did its job |
| 3 | `ndcg_at_k` in `encode.py` carries a vestigial `pos = {d: i for …}` dict and a dead `_ = pos` | Reading the diff before commit | None — dead code, no effect on any number | Left in place deliberately would be wrong; noted here, removed |

## Near-miss (not an error, but the interesting one)

`rotation-decorrelation` already had a `jina_scifact_corpus.npy` — same corpus,
same embedder — cached in `claude-container-layers` releases. Reusing it would
have skipped the encode entirely and been **wrong**: it is corpus-only, and
nothing anywhere records which `dim` / `max_length` / prefix / pooling produced
it. Had it happened to be 256-dim, the shape check would have passed and the
mismatch would have been undetectable — a silently non-comparable matrix
delivered against an issue whose entire premise is comparability with the
2026-07-08 codec eval.

Direction: this is the failure mode where prior art makes things *worse*. The
repo's standing instruction is to search for prior work before building; the
correct outcome here was to find it and then decline to use it. "Found prior art"
and "may reuse prior art" are different conclusions, and the provenance gap is
what separates them.

## What was not checked

- **No fp32-vs-q4 fidelity number was produced here.** The 0.975 per-doc cosine
  is inherited from the mirror's `PERFORMANCE.md` on a different corpus
  (NFCorpus/muninn), not re-measured on SciFact. Producing it would mean
  downloading the 847 MB fp32 export and encoding twice; out of scope for this
  issue, and flagged so nobody reads 0.7152 as a q4-fidelity result.
- **No third-party scorer.** `pytrec_eval` does not build on this container, so
  nDCG@10 rests on two implementations written in the same session by the same
  author. They agree to < 1e-9 and the negative controls collapse correctly,
  which rules out arithmetic slips but not a shared misreading of the metric's
  definition.
- **Single machine, single run.** No seed variance — the encode is
  deterministic (bit-identical across batch orderings and batch sizes, measured),
  so there is nothing to average, but that is a claim about this ONNX
  CPU path only.
