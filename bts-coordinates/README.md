# bts-coordinates

Does an LLM-named, growing coordinate system beat flat dense similarity at
surfacing a cross-field prior-art target? Transplant of the dynamic-support
mechanism in Large Discovery Models (arXiv:2608.15669 §4.2) onto the
between-the-spokes problem.

**Negative.** The growing-coordinate arm shows no advantage over its own frozen
ablation (one win each across four informative comparisons, sign test p = 1.0),
and the first pass's headline — a blind-named axis ranking the target 5th of
1101 — did not survive its null control: twelve *random* pool titles used as
axes give a median best-of-12 rank of 10. Best-of-k was an order statistic.

What survives is narrow: averaging over a spread of short named probes beats
both the raw query and random probes in all five configurations. That is
multi-probe query expansion, which `../METHODS.md` records as losing three
times previously in this repo.

Separately, and upstream of everything tested here: a subagent given the
problem with all cross-field vocabulary stripped, using **zero tools**,
returned the target paper's verbatim title as its top search query in under two
minutes. The ms13 campaign took until phase 7 to reach the same connection.

- [`PLAN.md`](PLAN.md) — pre-registration, written before any data, including
  the null that turned out to hold and the adversarial pass that killed the
  headline.
- [`RESULTS.md`](RESULTS.md) — full writeup, controls, limitations, and the
  infrastructure findings that contradict the July 2026 record.
- `queries.py` — case texts, including the leak found in the inherited test case.
- `fetch_corpus.py`, `embed.py`, `arms.py`, `run_*.py`, `null_control.py`.
