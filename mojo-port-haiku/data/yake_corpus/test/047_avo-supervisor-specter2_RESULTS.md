# avo-supervisor-specter2

Origin: [claude-workspace#233](https://github.com/oaustegard/claude-workspace/issues/233) — "Assess AVO-style supervisor loop in CCotw". That issue's verdict was **go** on the smallest version. This run is the first real search driven off that infrastructure.

**Goal:** maximize recall@10 for remex quantization on real SPECTER2 embeddings (768-d scientific-paper vectors) at ≥7.8x compression (bits ≤ 4), 30-candidate budget.

## Method

- **Data:** `specter2_nlp_broad.npy` (10,000 × 768 float32, allenai/specter2_base embeddings), pulled from `oaustegard/claude-container-layers` release `specter2-nlp-broad-10k`.
- **Split:** rows 0–9799 = corpus (9,800 vectors), rows 9800–9999 = held-out queries (200 vectors).
- **Ground truth:** exact inner-product top-10 per query, computed once and cached to `specter2_truth_top10.npy` next to the fitness script.
- **Fitness function:** [`fitness_specter2.py`](https://github.com/oaustegard/claude-workspace/blob/main/.supervisor/fitness_specter2.py) in the hub repo — one `remex.Quantizer` config in, recall@10 out, ~2–14s per call depending on strategy.
- **Compression gate:** bits ≤ 4 enforced in the script (32/4 = 8.0x ≥ 7.8x required).

## Infrastructure note — the supervisor loop did not actually run

The plan (per #233's "smallest version worth building") was to arm `.supervisor/run.json` and let the hub's `scripts/supervisor_stop.py` Stop hook drive candidates by blocking each stop. That script is real, merged in [claude-workspace#237](https://github.com/oaustegard/claude-workspace/pull/237), and matches the schema this run was configured against.

It didn't drive this run. This session is a **child session** spawned from another Claude Code session (not a CCotw/Muninn hub boot), and its own Stop hooks are unrelated harness scaffolding (`~/stop-hook-git-check.sh`, `~/stop-hook-reply-gate.py`) — nothing here reads `.supervisor/run.json` or blocks on it. Arming the file would have been a no-op.

Given that, the 30 candidates below were run directly in one session via a small driver ([`run_candidates.py`](run_candidates.py)) that applies the same plateau rule the hook would have (window=4, min_delta=0.001, max_strategy_switches=3) and the same ledger schema, but under my own control rather than a Stop-hook block. Functionally equivalent search; different mechanism. Whether the hook mechanism itself works as designed remains what #233's own verdict comment already tested (it does, in the hub environment) — this run doesn't re-test that, it just couldn't use it here.

## Results

**Best: bits=4, rotation=haar, seed=2026, two-stage rescore (candidates=300) → recall@10 = 0.7630** (8.00x compression)

vs. **baseline bits=4/rht/seed=42 (the library default) → recall@10 = 0.7510** — a +0.012 (1.2 point) improvement, entirely from seed selection plus a marginal two-stage rescore gain.

| Rank | Config | Strategy class | R@10 | Compression |
|---|---|---|---|---|
| 1 | bits=4 haar seed=2026 two-stage(300) | search-strategy | **0.7630** | 8.00x |
| 2 | bits=4 haar seed=2026 IVF lsh n_bits=4 nprobe=16 (=flat) | ivf-coarse-tier | 0.7625 | 8.00x |
| 3 | bits=4 haar seed=2026 (flat) | quantization-config | 0.7620 | 8.00x |
| 4 | bits=4 haar seed=123 | quantization-config | 0.7580 | 8.00x |
| 5 | bits=4 haar seed=42 | quantization-config | 0.7565 | 8.00x |
| — | bits=4 rht seed=42 (library default, baseline) | quantization-config | 0.7510 | 8.00x |
| — | bits=3 haar seed=42 | quantization-config | 0.6360 | 10.67x |
| — | bits=2 rht seed=42 | quantization-config | 0.4975 | 16.00x |

By strategy class (30 candidates total):

| Class | n | best R@10 | mean R@10 |
|---|---|---|---|
| quantization-config | 18 | 0.7620 | 0.7153 |
| search-strategy (two-stage rescore) | 6 | 0.7630 | 0.7365 |
| ivf-coarse-tier | 6 | 0.7625 | 0.7528 |

**Strategy switches: 2 of 3 allowed** (plateau after search-strategy sweep at candidate 15, plateau again after ivf-coarse-tier sweep at candidate 20; final quantization-config re-sweep at candidates 21–25 found the seed=2026 improvement and never re-plateaued before budget ran out).

## What moved the number, and what didn't

- **Rotation seed matters more than rotation construction.** `haar` vs `rht` measures indistinguishable on remex's own synthetic benchmarks (±0.0013 per the README), but on this real 9,800-paper corpus, seed choice alone spans 0.7285–0.7620 (haar) — a wider range than the haar/rht gap at any fixed seed. This is consistent with a small, non-isotropic real corpus being more sensitive to the specific random rotation draw than remex's synthetic Gaussian test data is.
- **Two-stage rescoring barely helps at this corpus size.** Candidate pool sweeps from 100–800 on the best config move R@10 by ~0.001–0.006 — noise-level. Two-stage exists to save memory/latency on large corpora, not to raise the recall ceiling on 9,800 vectors, and that's exactly what it did here.
- **IVF coarse-tier tops out at flat-scan recall, as documented.** `nprobe = 2**n_bits` (visiting every cell) reproduces the flat-scan score almost exactly (0.7625 vs 0.7620 flat, same seed) — remex's README says this is exact in that limit, and it held. Below full nprobe, recall drops with pool size as expected (n_bits=6/nprobe=32 at ~50% cells: 0.7385). At a 9,800-vector corpus, IVF has nothing to win: the README's own guidance says it earns its keep only above ~10M vectors, which this run reconfirms rather than contradicts.
- **bits=3 and bits=2 aren't competitive at this recall target.** 3-bit tops out around 0.636 (10.67x compression), 2-bit around 0.50 (16x) — both far below the ≥0.75 the 4-bit configs reach. If the actual requirement is "≥7.8x compression, whatever recall that buys," bits=4 is the only band worth searching in; the budget would have been better spent entirely inside quantization-config and search-strategy at bits=4 rather than spending 6 candidates each on bits≤3 and IVF. Noted rather than re-run, since the budget was fixed going in.

## Throughput

30 candidates in a 334s wall-clock span (227s of actual fitness compute, rest is driver/ledger/commit overhead) — roughly 320 candidates/hour at this pace. The fitness function is not the bottleneck for this domain: at 2–14s per candidate, a 30-candidate budget is a matter of minutes, not the "20 minutes per direction" AVO reported for GPU-kernel optimization. The real constraint on candidate count here was the search space (seed values, config combinations), not wall clock.

## Anomalies

- One transient GitHub 503 ("authorization temporarily unavailable") on the first push after candidate 5; resolved on retry ~5s later. Consistent with the hub CLAUDE.md's note that GitHub-adjacent 5xx are typically transient infra blips, not credential problems.
- A `__pycache__/` directory was accidentally committed in the first push (candidates 1–5) and removed in the next commit with a `.gitignore` added. No data loss, just a one-commit cleanup.
- The supervisor-loop infrastructure mismatch above (real hook, wrong session) was the only outcome that changed how this run was actually executed vs. how it was specced.
