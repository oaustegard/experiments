# Errors caught during this run

None caught in-run — the harness, splits, and incumbents were inherited from
`bekko-embedding-bench` unchanged, and both cross-run anchors landed exactly
(jina blog/code R@10 0.631/0.978 to the third decimal; bekko-a8m 0.575/0.888;
sklearn corpus 11,380 AST chunks over 674 files).

Risks that were checked rather than assumed, recorded because the base rate
of a body of work is worth knowing:

- **The parent experiment never pinned its sklearn commit.** The corpus was
  rebuilt at `7cb1868aa` (last commit before the parent's 2026-08-05 run) and
  only accepted because the chunk/file counts and the two incumbent anchors
  reproduced exactly. Had they drifted, every cross-run comparison here would
  have been silently non-comparable; the anchors are the check.
- **The ONNX exports omit the sentence-transformers Dense module in
  `last_hidden_state` form.** The graphs also ship a `sentence_embedding`
  output with pooling + Dense baked in; the hand-rolled path was validated
  against it (cos = 1.000) before any benchmark ran. Skipping that check and
  pooling without the 384→1024 projection would have produced a plausible
  but wrong 384-d model.
