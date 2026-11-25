# Delegation ledger

Oskar asked for mechanical work to go to cheaper models. Kernel design, the reference, verification
and the write-up stayed with the coordinating session. Prices: gemini-4-hydrogen $1.50 / $7.50 per M
input / output tokens (thinking billed as output).

| # | job | model | tokens in / out (+thinking) | cost | verdict |
|---|---|---|---|---|---|
| 1 | `pack.py` converter from a written format spec (regions, i8 and MatMulNBits-q4 layouts, manifest schema, `dequant_state_dict`) | gemini-4-hydrogen, thinking low | 2,105 / 5,006 (+2,756) | $0.061 | accepted; removed one unused variable (ruff F841), compacted the per-layer loop. Byte layout checked by the WGSL engine reading it and by the dequantized-weights reference matching the pack |
| 2 | `scifact.py` encode/score with checkpoints, nDCG@10, paired bootstrap | gemini-4-hydrogen, thinking low | 1,331 / 4,336 (+1,480) | $0.046 | accepted unchanged apart from formatting and a module docstring |
