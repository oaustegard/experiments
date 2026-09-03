# Delegate brief — potion-code-16M-v2 vs bekko-a8m on mini-CTXBench, plus output-quantization ladder

Parent session: https://claude.ai/code/session_015CSL4eo3Ap5VJfKVzv1qJX (Muninn on CCotw, 2026-09-03).
Provenance: memories cabf6b3d (what potion-code-16M-v2 is), 4496fa25 (zvec-grep eval),
468f90ab (bekko-bench), METHODS.md line ~1590 ("re-measure the bit ladder per encoder").
Prior-art check already done by the parent: Model2Vec natively offers int8 TABLE
quantization and PCA-ordered dimension truncation (`StaticModel.from_pretrained(path,
quantize_to="int8", dimensionality=N)`); HF ships no other variants. What is NOT
measured anywhere is (a) potion vs bekko on the n=59 file-discovery task and (b) how
potion's OUTPUT vectors survive remex/remax quantization. Both are the deliverable.

Everything below runs in THIS container. No network writes. Do not open PRs, issues
or comments. Report back in your final message; the parent commits.

## Fixed facts (do not re-derive)

- Harness: `/home/user/experiments/bekko-embedding-bench/` — `instances.json` (n=59),
  `scripts/chunk.py`, `scripts/bekko.py`, `scripts/eval_search.py`, `scripts/encode_corpus.py`,
  `scripts/run_code_quant.py` (the ladder to mirror; read it first, it is 140 lines).
- Corpus: scikit-learn `sklearn/` subtree. Clone (public, anonymous read works):
  `git clone -q https://github.com/scikit-learn/scikit-learn /home/user/sklearn-bench && git -C /home/user/sklearn-bench checkout -q 7cb1868aac973906fe5293485133d873c0fe6c42`
  (newest instance's merge commit, PR #34645). Give the clone a 10-minute timeout; it
  is a full-history clone because a shallow one may not contain that sha — try
  `--depth 1` first then `fetch --depth=2000` if checkout fails. Export
  `BEKKO_BENCH_REPO=/home/user/sklearn-bench`. `rg` must be on PATH (it is).
- Build chunks: `cd bekko-embedding-bench && python3 scripts/chunk.py` -> `chunks_ast.json`
  (expect ~11k chunks / ~680 files; report the exact numbers). Report gold coverage:
  fraction of gold files across the 59 instances that exist in the corpus.
- bekko-a8m: `scripts/bekko.py` expects `/home/user/models/bekko-a8m/{onnx/model.onnx,tokenizer.json,config.json}`.
  Fetch from `https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/resolve/main/<file>`
  with `curl -sSL` (works from this container; tokenizer.json is 34 MB, model.onnx 124 MB).
  Then `python3 scripts/encode_corpus.py --variant a8m --mode ast --threads 4` -> `vecs_ast_a8m.f32`
  (~10 min). Run it detached with `nohup ... > encode_a8m.log 2>&1 &` and do potion work
  meanwhile; check with ONE `tail` per turn, never a sleep loop.
- potion-code-16M-v2: already downloaded at `/home/user/experiments/.cache/potion/`.
  `pip` has `model2vec` 0.9.0. Usage:
  `from model2vec import StaticModel; m = StaticModel.from_pretrained("/home/user/experiments/.cache/potion")`;
  `m.encode(texts, max_length=512, batch_size=1024)` -> (N,256) float16, L2-normalized.
  Cast to float32 before any matmul. Use max_length=512 for parity with bekko (512).
  Verified: 2000 chunks encode in 0.7 s.
- remex: `pip` package 0.6.0 (`remex.Quantizer(d=, bits=, seed=0)`, `.encode/.decode`).
- remax: source at `/home/user/oaustegard/remax/src` — `sys.path.insert(0, ...)`;
  `from remax import StackedSignBitQuantizer`.
- Scoring helpers: `eval_search.extract_identifiers, arm_rg, recall_at, rrf`, and
  `run_code_quant.file_rank / hamming_rank`. Queries are `title + "\n" + body`.

## What to produce

Write `/home/user/experiments/potion-code-quant/run.py` (one script; may import from
`../bekko-embedding-bench/scripts`) and `results.json` (list of rows like
results_code_quant.json, plus an `encoder` field), and `RESULTS.md`.

Cells, all scored as file-level r@5, r@10 and RRF(rg + dense) r@10 on the n=59 task,
same corpus and same rg baseline for every row:

1. **rg baseline** (once).
2. **bekko-a8m fp32 d=384** (reference; the harness's own number was r@5 0.595 at
   n=59 — say whether you reproduce it on this checkout).
3. **potion fp16 d=256** as shipped.
4. **potion dimension truncation** d=128, 64 via `dimensionality=` at load (PCA-ordered
   by construction — state that the load-time path and slicing the 256-d output give the
   same ranking, or that they don't).
5. **potion native int8 table** (`quantize_to="int8"`), d=256 — encoder-side quantization.
   Report cosine of its outputs to the fp16 outputs (mean/min over corpus).
6. **potion output remex** 1/2/4-bit at d=256 (encode corpus, decode, renormalize, dot
   with float query — exactly as run_code_quant.py does).
7. **potion output remax** k=1/2/4/8 at d=256, Hamming ranking.
8. **bekko-a8m output remex 2-bit d=384 and remax k=1 d=384** (the two deployed
   settings: xr uses remex 2-bit, remax_kb uses 1-bit) so potion cells have a same-run
   comparator at matched bytes: potion remex 2-bit d=256 = 64 B vs bekko remex 2-bit
   d=384 = 96 B; potion remex 4-bit d=256 = 128 B; potion remax k=1 = 32 B.
9. **Timing**: corpus encode wall-clock for a8m (from the log) and potion; per-query
   encode latency for both, median over the 59 queries, 1 thread if you can pin it
   (`OMP_NUM_THREADS=1`), otherwise say 4.
10. **Stratum split**: report r@5 for identifier-poor vs identifier-rich instances
    (poor = `extract_identifiers` returns empty) for rg, bekko fp32, potion fp16.
    The open question from cabf6b3d is whether a static model hurts the poor stratum.

For paired significance use the same method as `scripts/bench_significance.py`
(read it; sign test / paired bootstrap on per-instance r@5) on: potion-fp16 vs
bekko-fp32; potion remex 2-bit vs potion fp16; potion remax k=1 vs potion fp16.
At n=59 most gaps will be non-significant — say so with the p-values rather than
picking a narrative.

## RESULTS.md rules

Lead with the numbers table. Facts as short sentences. No "interestingly", no
"notably", no rhetorical questions, no bullet-point summary at the end, no
"conclusion" header. State what did not replicate as a plain fact. Report the box
(`nproc`, RAM) once. Note anything that failed and how you worked around it.
Do not commit; the parent reviews and commits.

Budget: one pass. If a8m encoding is still running when everything else is done,
finish potion rows, write results, then wait for a8m with a single detached
`tail -f` check per turn. Total expected wall time ~20-30 min.
