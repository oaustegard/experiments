# ERRORS — emeans-pooled-residual

1. **SciFact bench killed at the 2 h background-job cap** (94 of 100 rows). Every unit checkpoints to
   `results_<corpus>.jsonl` and `.work/`, so a relaunch finished the remaining six rows; no rows were re-run or
   lost. Budget long benches per corpus, not per run.
2. **Prediction G3 rested on a 20k-row smoke test.** At n = 20k, k = 64, d = 32 torch `Emeans` and
   `NumpyEmeans` took 0.50 vs 0.48 s, so PLAN.md predicted torch CPU would not be faster. At n ≥ 100k and
   d ≥ 128 it is 2–10× faster. A smoke test checks that code runs; it says nothing about scaling.
3. **The first byte frontier took the best of four trainers per point.** One Lloyd-25 arm (SciFact pf2
   K = 4,096, 1 bit, 0.769) sat 0.011–0.013 above the three other trainers at the same setting, and a max over
   sources put it on the frontier. `analyze.py` now builds the frontier from the pre-chosen e-means-2 arms and
   the single mean only.
4. **`colour_bench.py` ran once by accident** when a profiling snippet imported it (the script has no
   `__main__` guard). The run was complete and identical in setup, so `results_colour.json` holds both images.
