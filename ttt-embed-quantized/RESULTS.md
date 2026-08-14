# SciFact encoded with jina-v5-nano q4 — the committed corpus matrix

**Status: done.** This is a *corpus-encode artifact*, not a hypothesis test. The
deliverable is `data/{Dm.npy,Q.npy,meta.json}` so that claude.ai — where the
same encode is 45–85 min and detached jobs die silently after ~100 s — never has
to pay for it again. ([issue #33](https://github.com/oaustegard/experiments/issues/33))

The experiment these feed (TTT-Embed task vector under an oracle teacher, swept
through remex 8/4/2/1-bit and remax k=8/4/2) is *not* here — it is seconds of
numpy on these matrices and runs elsewhere. Paper: arXiv 2608.12569.

## Headline

| | |
|---|---|
| **fp32 nDCG@10** (`Q @ Dm.T`, BEIR qrels) | **0.7152** |
| Expected band (issue) | 0.60–0.72 — **inside, near the top** |
| Recall@10 / Recall@100 | 0.8346 / 0.9483 |
| Wall clock | **14.7 min**, 4 vCPU |
| Throughput | 5.9 docs/s sustained (69 queries/s) |
| Artifact size | 5.3 MB (`Dm.npy`) + 0.3 MB (`Q.npy`) + 85 KB (`meta.json`) |

`Dm` is (5183, 256) float32, `Q` is (300, 256) float32, both L2-normalized to
within 1.2e-7. `meta.json` carries `doc_ids` / `q_ids` / `qrels` as strings,
aligned to row order, plus the full encoder provenance.

## Settings (pinned, unchanged from the 2026-07-08 codec eval)

Encoder `oaustegard/jina-v5-nano-mirror` release `v5-nano-8a7f00aa`, asset
`model.q4.onnx`, SHA256 `b8b18777…2552e15` verified against the mirror README
(169,736,452 bytes, retrieval LoRA merged, int4 MatMulNBits). `dim=256`
Matryoshka truncation, `max_length=384`, doc text `title + ". " + text`,
prefixes `"Document: "` / `"Query: "`, last-token pool (`mask.sum(-1) - 1`),
truncate-then-L2-normalize, fp32.

The pool/prefix/truncate/normalize order is lifted from the mirror's own
torch-free `scripts/embed_onnx.py`; the single deviation is the q4 asset in
place of that loader's pinned fp32 `model.onnx`.

## Verification

`recheck.py` reloads the committed artifact cold and recomputes nDCG@10 through a
deliberately different code path — `sorted()` over Python floats instead of
`np.argpartition`, an explicit `math.log2` DCG loop instead of a vectorised
discount array. It reproduces **0.715232 vs 0.715232** (< 1e-9), and 17/17
checks pass including shape/dtype, normalization, id alignment, no collapsed
duplicate rows, and qrels/corpus containment.

Two negative controls confirm the scorer can go red: shuffling the doc-id
mapping drops nDCG@10 to **0.0031**, shuffling query rows to **0.0040**. A
sanity check that cannot fail is not evidence.

`pytrec_eval` — the obvious third-party cross-check — has no wheel and needs a
compiler; the build fails on this container. The hand-rolled second
implementation is the substitute, not an addition to it.

## What broke / what surprised

**The issue's HF CDN warning did not reproduce here.** The spec says HF
load-balances its LFS redirect between `us.gcp.cdn.hf.co` (allowlisted) and
`us.aws.cdn.hf.co` (not), and to retry until it lands on GCP. All three files
landed **first try, on `us.aws.cdn.hf.co`**, at exactly the sizes in the HF
tree. This matches `bekko-embedding-bench`'s finding that `*.cdn.hf.co` egress
works from CCotw and contradicts it *only* for the claude.ai container, where
the same host was refused on 2026-08-04. The retry loop is kept in `encode.py`
because the artifact is meant to be reproducible from claude.ai too — it just
never fired here. **Allowlist state is per-environment; do not carry a refusal
from one container into a spec for another.**

**The encoder this issue pins is officially superseded.** The mirror's own
`PERFORMANCE.md` and `METHODS.md` both say the authors' upstream q4
(`jinaai/…-retrieval → onnx/model_q4.onnx`, 138 MB) beats this repo's
`model.q4.onnx` on every axis — smaller *and* more faithful to fp32 (NFCorpus
cosine 0.976 vs 0.974, nDCG@10 0.4291 vs 0.4250). Using it anyway is correct
here and deliberate: the entire point is that the 2026-07-08 codec eval's
fidelity numbers carry over, which requires bit-identical settings *and*
weights. Comparability beats absolute quality for an artifact whose job is to be
compared. Flagged rather than silently "improved".

**27% of documents hit the 384-token cap** (1397 / 5183; doc tokens mean 294,
median 300, p95 384). That is spec-mandated, not a defect, but downstream work
reading these vectors should know that better than a quarter of the corpus is
encoded from a prefix. Queries are nowhere near the cap (mean 22, max 56).

**Prior art exists and was not reusable.** `rotation-decorrelation` already
carries a `jina_scifact_corpus.npy` for exactly this corpus and embedder, cached
in `oaustegard/claude-container-layers` releases. It was *not* reused: it is
corpus-only (no queries, no qrels), and nothing records whether it used this
dim / max_length / prefix / pooling combination — an unverifiable settings match
is worse than a 15-minute re-encode when the whole deliverable is
"comparable to the codec eval".

## Cost

14.7 min wall clock on 4 vCPU; ~10 min of that is the document pass. Model
download 170 MB, dataset 4.5 MB. No API spend — everything is local ONNX CPU
inference. The claude.ai estimate this issue exists to avoid was 45–85 min on
1 core, which the 5.9 vs <2 docs/s gap corroborates.

## Reproduce

```bash
python3 encode.py                 # ~15 min on 4 vCPU; resumable, checkpoints every 40 batches
python3 recheck.py                # seconds; re-scores the committed artifact, exits non-zero on drift
python3 encode.py --parity-check  # confirms length-sorted batching is bit-identical
```

`encode.py` fetches the model (SHA256-verified) and dataset itself; `data/raw/`
and `checkpoints/` are gitignored as regenerable. The `.npy` files are committed
deliberately — they *are* the deliverable.

## Note on batching

Batches are formed after sorting by token length. Padding is masked out of both
attention and the pool index, so this is numerically inert, and `--parity-check`
confirms it: **max abs difference 0.0** — bit-identical, not merely close —
against unsorted batching, and identical again across batch size 8 vs 16.
Measured speedup on a random 240-doc sample is **1.37x** (4.3 → 5.8 docs/s).
Free, but smaller than the 2x that "halve the padding" intuition suggests, so it
is reported measured rather than assumed.
