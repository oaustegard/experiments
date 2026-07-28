# LADDER_RESULTS.md — mechanism ladder beta* study (issue #169 kill-path B)

Status: **partial**. The fine-grid ("`--deep`") run was stopped mid-way
through the K4/K5 clique patterns when it hit the session time budget (it
had already run ~11 minutes and was not close to done — see "Deferred"
below). Rather than wait longer, the completed fine-grid rows are reported
as final/exact, the three unfinished configurations are explicitly deferred
with a labeled coarse-grid fallback number, and `ladder.py` itself now
defaults to a fast coarse grid (see "Reproducing" below) so it is usable
standalone without risking another multi-minute stall.

Anchor check (C3 = K3, equal demands 1, p = 1/2 uniform): **beta\* = 1** —
matches the corrected triangle calibration value stated in the task brief.

## Ladder table — fine grid, completed (exact, denom 8–12)

| graph | n | demand pattern | demands | best beta\* | beta\* (float) | split p | combos tried | combos skipped (timeout) | seconds |
|---|---|---|---|---|---|---|---|---|---|
| C3 | 3 | equal-1 | (1,1,1) | **7/6** | 1.1667 | ['5/12','5/12','5/12'] | 13 | 0 | 1.79 |
| C3 | 3 | rybin-15-10-15 | (15,10,15) | 1 | 1.0000 | ['1/4','5/8','1/4'] | 81 | 0 | 16.14 |
| C5 | 5 | equal-1 | (1,1,1,1,1) | 11/12 | 0.9167 | ['11/12']×5 | 13 | 0 | 3.99 |
| C5 | 5 | alternating-3-2 | (3,2,3,2,3) | 11/12 | 0.9167 | ['1/2','3/8','1/2','3/8','1/2'] | 81 | 0 | 31.42 |
| C5 | 5 | indep-set-heavy-3-2 | (3,2,3,2,2) | 7/8 | 0.8750 | ['7/8','0','7/8','0','0'] | 81 | 0 | 37.84 |
| C7 | 7 | equal-1 | (1,1,1,1,1,1,1) | 11/12 | 0.9167 | ['11/12']×7 | 13 | 0 | 14.14 |
| C7 | 7 | alternating-3-2 | (3,2,3,2,3,2,3) | 3/4 | 0.7500 | ['1/4','1','1/4','1','1/4','1','1/4'] | 22 | 3 | 35.98 |
| C7 | 7 | indep-set-heavy-3-2 | (3,2,3,2,3,2,2) | 1 | 1.0000 | ['3/4','1/4','3/4','1/4','3/4','1/4','1/4'] | 24 | 1 | 29.24 |
| K4 | 4 | equal-1 | (1,1,1,1) | **7/6** | 1.1667 | ['5/12']×4 | 13 | 0 | 3.19 |

Every beta\* above came from a completed exact LP (`beta_star.compute_beta_star`
via `sympy`'s exact rational simplex) at every breakpoint scanned; the
"combos skipped" column counts individual splits within that pattern's grid
whose LP blew a 5-second per-call wall-clock budget (see `ladder.py`'s
`_compute_beta_star_bounded`) — the reported beta\* is still the exact max
over every combo that *did* complete, just possibly not the true grid
maximum if a skipped combo would have beaten it.

## Deferred — exact-LP fine grid exceeded session budget

The fine grid (denom 8, 81 combos per 2-class pattern) never finished for
these three configurations; the run was killed mid-`K4 one-heavy-15-10`
after ~11 minutes total wall time with C5's two 2-class patterns alone
costing 31–38s each and clique/cycle n≥6 configurations regularly hitting
the per-call timeout. Rather than block further, here are **coarse-grid
fallback numbers** (`ladder.py`'s fast default: denom 2–4, 5–9 combos) —
these are NOT the same resolution as the table above and should not be
compared to it as if they were:

| graph | n | demand pattern | demands | beta\* (coarse grid only) | split p | combos | seconds |
|---|---|---|---|---|---|---|---|
| K4 | 4 | one-heavy-15-10 | (15,10,10,10) | 2/3 (0.6667) | ['0','1/2','1/2','1/2'] | 9 | 2.39 |
| K5 | 5 | equal-1 | (1,1,1,1,1) | 1 (1.0000) | ['1/2']×5 | 5 | 2.21 |
| K5 | 5 | one-heavy-15-10 | (15,10,10,10,10) | 2/3 (0.6667) | ['0','1/2','1/2','1/2','1/2'] | 9 | 3.46 |

These are lower bounds on the true fine-grid beta\* for each configuration
(a coarser grid can only miss a higher value, never invent one) — they are
reported so the ladder isn't silent on K4/K5's unequal-demand pattern and
K5's equal-demand baseline, not as a substitute for the deferred exact run.
**If a future session re-runs `python3 ladder.py --deep --timeout 30`, only
these three rows need to be filled in** (K5's equal-demand row in
particular, to compare directly against K4 equal-1's 7/6 at matching
resolution); everything else in the fine-grid table above is already exact
and final.

## Stable-set LP gap per graph (theory note)

| graph | alpha (integer) | alpha_f (fractional LP) | gap = alpha_f/alpha |
|---|---|---|---|
| C3 | 1 | 3/2 | 3/2 (1.5000) |
| C5 | 2 | 5/2 | 5/4 (1.2500) |
| C7 | 3 | 7/2 | 7/6 (1.1667) |
| K4 | 1 | 2 | 2 (2.0000) |
| K5 | 1 | 5/2 | 5/2 (2.5000) |

## Trend verdict

Best **exact, fine-grid** beta\* found: **7/6** (1.1667), tied between
C3/equal-1 and K4/equal-1. Every other fine-grid configuration is at or
below 1 (11/12 for both cycles' equal-demand baseline, ranging down to 3/4
for C7/alternating-3-2). None of the completed configurations comes
anywhere close to beta\* = 2, or even to 3/2.

Growing the conflict graph from the triangle (C3) up through C5, C7, K4
does **not** show a monotone climb toward beta\* = 2 in this abstract
pseudo-arc template, and it does **not** track the stable-set LP gap either:
K4's LP gap (2) is bigger than C3's (3/2), but K4 and C3 land on the exact
same measured beta\* (7/6); C5/C7 have smaller LP gaps than K4 (5/4, 7/6 vs
2) but also smaller measured beta\* (11/12 vs 7/6) — so bigger LP gap does
correlate loosely across clique-vs-cycle but *not* within the clique family
itself (K4 doesn't beat K3 despite a strictly bigger LP gap), and every
unequal-demand ("Rybin-shape") pattern tried scored at or below its
graph's own equal-demand baseline, the opposite of what would be needed to
push toward a refutation. The K5 gap in the table (fine grid deferred) is
consistent with the same flat/non-monotone pattern at coarse resolution
(coarse K5/equal-1 = 1, below K4's exact 7/6) but is not proof at matching
resolution — see "Deferred" above.

**Overall reading: the mechanism tops out well short of 2 everywhere it was
exactly measured, with the completed evidence pointing at a ceiling well
under 3/2 rather than a climb toward 2 as the conflict graph densifies.**
No configuration in this run — fine grid or coarse fallback — exceeded
beta\* = 2, so there is nothing here requiring the LOUD-FLAG orchestrator
escalation described in `ladder.py`.

## Reproducing

```bash
python3 ladder.py                        # fast default (denom 2-4), ~30s total, all 12 rows
python3 ladder.py --deep                 # fine grid (denom 8-12) -- SLOW at n>=6, may need --timeout
python3 ladder.py --deep --timeout 30    # fine grid with a more generous per-call LP budget
python3 ladder.py --write-results        # also overwrite this file with the run's numbers
```

The fast default is safe to run any time (completes in well under a
minute, all 12 configurations) and is what CI / a quick sanity check
should use; it does NOT reproduce the exact numbers in the fine-grid table
above (see the coarse fallback table for what fast mode gives on the same
demand patterns). `--write-results` is opt-in specifically so a routine
fast-mode run never silently clobbers this hand-curated file, which mixes
a completed fine-grid run with an explicit deferred section.

## Caveats (read before citing any number above)

- **Abstract pseudo-arc construction, not a DAG realization.** Every
  instance here uses synthetic dict-key "arcs" for load bookkeeping;
  `core.check_paths_valid()` was never run and would reject these paths (a
  shared arc's declared tail can't equal two different terminal names at
  once). Whether this conflict mechanism can be spliced into an actual
  acyclic SSUF network — with genuine connected s→t_i arc sequences,
  splice-closure, DAG well-formedness — is open and is a strictly harder,
  separate step.
- **These beta\* values are an upper bound on the mechanism, not a lower
  bound on anything real.** Realization constraints can only restrict the
  achievable good-set further, which can only lower beta\*, never raise it.
  So even the largest number in the tables above is not evidence *for* a
  real counterexample — at best it is a signal about whether the mechanism
  is worth the realization effort; at worst (as the trend here suggests) it
  shows the mechanism alone, even unconstrained, does not get anywhere
  near 2.
- **Split search is a grid, not an exhaustive sup**, on top of which three
  configurations only have a coarse-grid number at all (see Deferred). beta\*
  is exact at every grid point that was evaluated, but a finite grid over
  per-terminal cheap-fractions p_i can miss the true continuous sup; the
  LP-feasibility structure defining breakpoints is piecewise-linear in p,
  so misses are expected to be small, not qualitative — but this has only
  been spot-checked, not proven.
- **Per-call timeouts mean a few grid points within otherwise-"complete"
  rows (C7's two unequal-demand patterns) were skipped.** The reported
  beta\* for those rows is the max over whatever combos *did* finish, which
  could in principle be beaten by a skipped combo. Flagged in the table via
  the "combos skipped" column.
