# Issue #76 build scripts — full-corpus `muninn.kb`

Reproduces the artifact published in
[oaustegard/muninn.austegard.com#136](https://github.com/oaustegard/muninn.austegard.com/pull/136).

## Inputs

- `oaustegard/muninn.austegard.com` checkout — corpus source (HTML under
  `blog/`, `perch/`, `scratch/`)
- `oaustegard/jina-v5-nano-mirror` checkout — torch loader (downloads
  ~440 MB of weight + adapter assets on first use, cached under
  `~/.cache/jina-v5-nano-mirror/`)
- `remax_kb` and `remax` installed: `pip install -r
  .spokes/remax_kb/requirements-build.txt && pip install -e
  .spokes/remax_kb`

The default paths assume both spokes are cloned under `.spokes/`.

## Steps

```bash
# 1. Extract chunks (sanity peek)
python3 experiments/muninn-kb-issue-76/extract_chunks.py .spokes/muninn.austegard.com --limit 2

# 2. Build the .kb (writes to .spokes/muninn.austegard.com/knowledge/muninn.kb)
python3 experiments/muninn-kb-issue-76/build_muninn_kb.py --batch-size 16

# 3. Run acceptance queries
python3 experiments/muninn-kb-issue-76/query_muninn_kb.py
```

End-to-end wall-clock: ~2.5 min on CCotw (4 CPU, 15 GB).

## Determinism

Pack params (`dim=256, k=8, seed=0`, jina-v5-nano-text retrieval +
last-token) are pinned in `build_muninn_kb.py` and match
`muninn-subset.kb` for direct comparability. Same corpus + same params =
bit-identical artifact across runs, modulo fp numerics under the
binarization threshold.
