# Error log — neomme-remex-quant

Errors found in this run, how each was caught, and which direction it pushed the
conclusion. Direction first: an error that makes a result look stronger is the
dangerous kind.

| # | Error | Caught by | Direction | Fixed |
|---|---|---|---|---|
| 1 | `neomme_quant.py` imported a `remax.packing.unpack` that does not exist in remax 0.2.0 (assumed from the module name) | First run of the self-test | None — import-time crash, no number produced | Yes — removed; the ±1 decode uses `np.unpackbits` directly |
| 2 | Smoke bench (`results_smoke.json`, 224 docs) ran on the *longest-first* checkpoints, so its bytes/doc (345 KB fp32 late) is 1.8x the corpus mean and its nDCG is over 12 queries | Known at design time; the smoke set is for exercising code paths | Would have overstated late-interaction storage cost by ~1.8x if quoted | Not a result — `results_smoke.json` is not committed and no number from it appears in `RESULTS.md` |
