# Delegation ledger

This study ran as one Claude Opus session. It kept design, verification and the
write-up, and sent mechanical or independent jobs to cheaper models: Gemini via
the Muninn `generate` tool (Muninn is Oskar's agent toolkit), and Claude subagents
(run inside the session's subscription, so no per-call price). Prices: Gemini
4 Hydrogen $1.50 / $7.50 per M tokens in / out (thinking billed as output).

| # | job | worker | tokens in / out | cost est. | outcome |
|---|---|---|---|---|---|
| 0 | Read arXiv 2611.09680 v2 and report, with quotes, what it says about value quantization, index bytes, top-k, eval settings, per-dataset scores | web-fetch subagent (Agent tool) | 111,677 total (split not reported) | n/a (subscription) | Accepted. Its first fetch's tail was a model summary that was wrong on Table 2; it caught that and re-read verbatim from an offset. Column mapping (Arg/FiQ/NFC/SciF) was its inference, checked against BEIR naming. |
| 1 | Black-box test suite for `compress.py` from a written spec, without seeing the implementation | gemini-4-hydrogen via Muninn `generate`, thinking low | 2,159 / 6,268 (4,059 text + 2,209 thinking) | $0.050 | Accepted with two test-side fixes: `np.isin` compared float32 outputs against float64 levels (cast added), and the Lloyd fixed-point check took the mean in float32 (now float64). All 11 tests pass against the implementation. |
| 2 | `plot.py`: per-dataset + macro Pareto panels of delta nDCG@10 vs bytes/doc, from the arms JSON schema | gemini-4-hydrogen via Muninn `generate`, thinking low | 1,054 / 3,007 (1,810 + 1,197 thinking) | $0.024 | Accepted after one edit: added the `table8` marker and label case for an arm added after it was written. Checked visually on the four-dataset output. |
| 3 | Fresh-reader review of REPORT.md with 5 spot-checks against results/ | Sonnet subagent (Agent tool) | 99,178 total | n/a (subscription) | Accepted: all 5 spot-checks passed; 6 interpretation issues fixed (paired CIs added for the scale and matched-byte claims, per-dataset caveats, terms defined). |
