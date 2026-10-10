# Errors — pplx-embed-v2-quant

What was wrong, how it was caught, and which way it pushed the conclusion.

## 1. Centered remax with an uncentered float query (harness bug, fixed in extra.py)

**What.** `bench.py`'s "remax k=1/2 (asym) centered" arms took sign codes of
(x − μ) and scored the rotated query q·R against them, uncentered. The docstring's
argument was that q·μ is constant across a query token's candidates, so MaxSim's
per-document max is unchanged. That holds for exact residuals. A sign code carries no
residual length, so the score is q·ŝ = μ·ŝ + (q − μ)·ŝ. With ‖μ‖ = 0.945 against
residual lengths around 0.3, the μ·ŝ term is noise that swamps the signal.

**Caught by.** The arms scored nDCG@10 0.061 and 0.062, under fp32's centroid-only
PLAID floor of 0.724. That is not a plausible codec result.

**Effect.** `extra.py` re-runs them with the query centered as well ((q − μ)·R·ŝ).
The rows that report remax centered with a float query come from there. The broken
rows stay in `results_scifact.json` under their original names and are marked as
superseded in RESULTS.md.

## 2. Chunk assembly OOM-killed (encode.py, fixed)

**What.** The resumed encode finished all 41 chunks, then died with exit 137 while
stitching them together. The loop read `z["T"][...]` from an `np.load`ed `.npz`
once per document. Each access decompresses the whole array again, and each
per-document slice kept its own full copy alive.

**Caught by.** Exit code 137 after the last chunk log line, with no `mv_tokens.npy`.

**Effect.** None on the numbers. Each array is now read once per chunk, and a run
whose chunks all exist no longer loads the model. The rerun assembled
1,559,411 tokens in 19 s.

## 3. Encode outlived the 2-hour background cap (expected, resumed)

The first encode was stopped at the 2-hour cap with 38 of 41 chunks done. Document
chunks are length-sorted, so the last ones took 380–500 s each. The run resumed from
the saved chunks. Total encode time was about 2 h 18 min on 4 vCPU.

## 4. remex centered mode + Matryoshka precision decodes at the wrong length (a remex bug, not this experiment's)

**What.** remex 1.1.0 `_centred_lengths` solves the stored length m against the
full-precision direction only; `decode(c, precision=p)` reuses it for the p-bit
direction. Centered 1-bit read from an 8-bit code scored 0.527 against 0.765 for a
direct 1-bit encode.

**Caught by.** `extra.py`'s nesting arms. Uncentered, nested 1-bit equals direct
1-bit (both are the sign bit), so a centered 0.24 gap had to be the length.

**Effect.** The bench's centered rows are direct encodes and unaffected. `renest.py`
shows that re-solving m per precision from the decoded ‖x‖ closes the gap at every
width with no format change. Fixed in remex `8e9134f` on main (unreleased): `_effective_norms` re-solves m per
precision, and `tests/test_centered_mode.py::TestNestedPrecision` guards it.
