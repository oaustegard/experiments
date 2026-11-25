# Delegation ledger

This directory puts the SparseUp v2 sparse retriever in a browser page (see REPORT.md). Oskar asked that
mechanical coding go to cheaper models than the Claude session running the work. That session kept the
GPU kernel design, the PyTorch reference (`ref.py`), the accuracy checks and REPORT.md, and wrote each
delegated job's spec. A job is accepted when its output passes that session's review and the checks
that use it. This table lists every delegated job so far.

Gemini runs through the Muninn `generate` tool. gemini-4-hydrogen costs $1.50 per million input tokens
and $7.50 per million output tokens, with its reasoning ("thinking") tokens billed as output. "Thinking
low" is its smallest reasoning budget. Cost = input × 1.5e-6 + (output + thinking) × 7.5e-6.

| # | job | model | tokens in / out (+thinking) | cost | verdict |
|---|---|---|---|---|---|
| 1 | `pack.py`, the converter from the Hugging Face checkpoint to the browser weight pack (8-bit and 4-bit formats, manifest, a function that reads the pack back as float weights) | gemini-4-hydrogen, thinking low | 2,105 / 5,006 (+2,756) | $0.061 | accepted with edits: one unused variable removed. The WGSL engine reads the pack correctly, and the reference run on weights read back from the pack is the int8 arm of the parity table |
| 2 | `scifact.py`, retrieval check on the SciFact benchmark: encode with checkpoints, score nDCG@10, paired bootstrap against fp32 | gemini-4-hydrogen, thinking low | 1,331 / 4,336 (+1,480) | $0.046 | accepted with formatting only |
| 3 | `web/index.html`, the demo page (query and documents in, expansion terms and ranking out), verified in headless Chromium | Claude Sonnet subagent (Agent tool) | 116,596 total, not split by direction | not itemized | accepted; relay logging, a build check and plainer wording added afterwards |
