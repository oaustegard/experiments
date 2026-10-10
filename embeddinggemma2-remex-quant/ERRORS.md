# Error log — embeddinggemma2-remex-quant

Errors found in this run, how each was caught, and which direction it pushed the
conclusion. Direction first: an error that makes a result look stronger is the
dangerous kind.

| # | Error | Caught by | Direction | Fixed |
|---|---|---|---|---|
| 1 | `bench.py` imported the mxbai harness with `from bench import ...`; run as `python3 bench.py`, that resolves to this directory's own `bench.py` | Synthetic smoke run (`.work/smoke`, 500 random docs) before the real encode finished | None: import-time failure, no number produced | Yes: the mxbai module loads by path through `importlib.util.spec_from_file_location` |
| 2 | The grid asked remex for 6-bit codes; remex supports 1, 2, 3, 4 and 8 bits only | Same smoke run: `ValueError: bits=6 is not supported` | None: crash, no number | Yes: 6-bit dropped from the grid |
| 3 | First RESULTS.md draft carried four prose errors: the NFCorpus 192 B matched-byte delta with its sign flipped (+0.0032 for −0.0032), "a binarised query costs 0.024–0.026 for every sign code" (remax pays 0.035–0.038), "every arm at d=128 sits under the frontier except at 16–48 B" (true for SciFact only), and 69 arms per corpus (63) | Re-reading each prose number against `report_*.txt`, `pairs.txt` and the result JSONs before writing `recheck.py` | The sign flip favoured d=768 over d=512 at 192 B; the remax line understated its sym loss. Neither changed a finding | Yes: corrected, and `recheck.py` now recomputes every quoted paired delta from the per-query npz |
