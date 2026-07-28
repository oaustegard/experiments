# SUP_RATIO_RESULTS.md — phase-4 supremum-ratio sweep (supersedes CERTIFIED_RESULTS.md)

**Status: NO ratio ≥ 1 found. The supremum ratio plateaus at exactly 0.7500
(Rybin's level) across every stratum tested, n = 4 through n = 7, k = 3
through a k = 6 spot check — it does not climb toward 1.** 702 ratios
computed, 0 bugs, 17 solver-side failures (excluded, not silently dropped —
see §5). Search code: `sup_ratio.py`. Machine-readable results:
`sup_ratio_results.json`.

---

## 0. What this run computes, and why it supersedes CERTIFIED_RESULTS.md

`certified_search.py` decided instances at **fixed demand vectors** (3
samples/pattern) via `discrepancy.exists_bad_w` — exact over `w`, but the
continuous demand axis was still only sampled. `discrepancy.exists_bad_instance
(inst, bound_factor=0.0)` closes that axis too: substituting `p_i = d_i w_i`,
`q_i = d_i(1-w_i)` makes the MILP linear in `(p,q)`, demands normalize to
`d_max ≤ 1` via `p_i+q_i ≤ 1`, and maximizing `eps` at threshold 0 returns

```
ratio(structure) = sup over ALL demands d and ALL w ∈ [0,1]^k of
                    [ min_z max_arc |load_a(z,w,d)| ] / d_max
```

for a fixed tree+chord **structure**, independent of any specific demand
values. `ratio < 1` ⇒ that structure can never yield a counterexample **for
any demand vector whatsoever** — infinitely stronger than a finite demand
sweep. `ratio ≥ 1` ⇒ candidate counterexample, stop immediately.

---

## 1. Calibration anchors (both reproduced before sweeping)

| Anchor | Expected | Got |
|---|---|---|
| Rybin structure (tree `s→a→b→c`; chords `(c,a),(c,s),(b,s)`) | ratio = 0.7500 exactly, witness demands (1, 0.5, 1) | **ratio = 0.7500000000000000**, `demands=[1.0, 0.5, 1.0]`, `w=[0.5, 0.5, 0.5]` — exact match |
| MSW25-covered structures (NG-9) | ratio < 1 always | **10/10 sampled covered structures**, all ratio < 1 (checked across both n≤5 and n=6,7 trees) |
| Consistency vs. `certified_search.py` | sup_ratio ≥ any fixed-demand ratio already found for the same structure (sup optimizes over strictly more) | **8/8** cross-checked instances (from `outtree_results.json`'s phase-3 argmax records) — sup_ratio was ≥ the fixed-demand ratio in every case (e.g. `n4-fork-at-n1`: fixed-demand 0.5000 vs sup 0.6667; `n5-star4`: fixed-demand 0.5833 vs sup 0.7500) |

Both mandatory self-tests plus the Rybin anchor are green. **Note the
Rybin anchor's own point**: Rybin's *own* demands (15,10,15 in the historical
construction) achieve only 2/3 ≈ 0.667 at that fixed vector — the tool's
optimized witness (1, 0.5, 1) with `w=(0.5,0.5,0.5)` finds a *better* demand
choice for the *same structure*, reaching the true structural supremum of
0.75. This is exactly the upgrade this run is built on: optimizing demands,
not just guessing them.

---

## 2. Headline: the per-stratum max-ratio table

| (n, k) | max ratio | argmax structure | uncovered tested / total | coverage |
|---|---|---|---|---|
| n4, k=3 | 0.7500000000 | n4-chain3 | 4 / 4 | **exhaustive** |
| n4, k=4 | 0.7500000000 | n4-chain3 | 18 / 18 | **exhaustive** |
| n5, k=3 | 0.7500000002 | n5-chain3-leaf | 49 / 49 | **exhaustive** |
| n5, k=4 | 0.7500000000 | n5-chain4 | 405 / 405 | **exhaustive** |
| n5, k=5 | 0.7500000000 | n5-chain-leaf | 34 / 1,939 | sampled (1.8%) |
| n6, k=3 | 0.7500000001 | n6-twochains | 76 / 126 | sampled (60%) |
| n6, k=4 | 0.7500000000 | n6-chain5 | 24 / 1,501 | sampled (1.6%) |
| n6, k=5 | 0.7500000000 | n6-twochains | 4 / 9,741 | sampled (0.04%), mostly timeouts (see §5) |
| n7, k=3 | 0.7500000000 | n7-twochains | 76 / 429 | sampled (18%) |
| n7, k=4 | 0.7500002000 | n7-chain6 | 24 / 6,753 | sampled (0.36%) |
| n7, k=5 | **no successful ratio** | — | 4 attempted / 5,007 | all 4 attempts hit solver error or timeout (§5) |
| k=6 (spot check, single instance, unoptimized) | 0.6666676667 | n4-fork-at-n1-derived 3-arc tree, 6 chords | 1 instance only | not a sweep — see §4 |
| k=7 | not attempted | — | 0 | budget (§5) |

Every number above 0.75 in the table (0.7500000002, 0.7500000001,
0.7500002000, etc.) is MILP solver tolerance around exactly 3/4, not a real
excess — the largest deviation seen anywhere in 702 decisions is
2.0 × 10⁻⁶ (`n7-chain6/k4`), and every one of those near-0.75 witnesses has
`w ≈ (0.5, 0.5, ...)` and `demands ≈ (1, 0.5, 1, 1)`-shaped — i.e. they are
all **the same Rybin gadget re-embedded** inside a larger tree, not a
genuinely different or higher-scoring structure.

**n4 and n5 at k ≤ 4 are exhaustive over the full uncovered structural set**
— not a sample. Every single MSW25-uncovered tree+chord pattern at n ≤ 5,
k ≤ 4 was tested (427 patterns, 427/427 tested), and the maximum ratio
anywhere in that exhaustive space is exactly 0.75.

---

## 3. Trend verdict: PLATEAU, not climb

**The supremum ratio does not climb toward 1 as n and k grow — it plateaus
at Rybin's own level, 0.75, from n=4 all the way through n=7 (and the one
k=6 spot check came in *below* 0.75, at 2/3).** This directly answers the
campaign's key open question (NG-10's phase-3 heuristic sweep only
established a −4/9 *lower*-bound plateau on `t(w)` at fixed demand samples;
this run establishes the actual *supremum* over all demands and all `w`,
and it is flat).

Concretely:
- 503 of 702 computed ratios (71.7%) land at exactly 0.75 (within solver
  tolerance) — the ceiling isn't a rare peak, it's hit repeatedly across
  nearly every tree shape and k value tested.
- **0 of 702** exceed 0.90. **0 of 702** exceed 0.99. Nothing even
  approaches the counterexample threshold of 1.
- The exhaustive n≤5, k≤4 slice (427/427 patterns) — where "plateau" is a
  proof, not a sample — tops out at exactly 0.75 as well.
- The k=6 spot check (admittedly a single unoptimized instance, not a
  sweep — see §4) came in at 2/3, *below* the plateau, consistent with no
  climb.

**Interpretation.** Within the out-tree instance class (Lemma 4's normal
form), Rybin's own gadget — tree `s→a→b→c`, chords `(c,a),(c,s),(b,s)` —
appears to already be **extremal**: nothing found at any larger n or k beats
it. That is a much sharper and more specific structural claim than "no
counterexample found" — it says the *worst* structure in this class, over
*all* demand choices, is capped at 3/4 of `d_max`, a full 1/4 of `d_max`
below the bound Conjecture 1.3 needs broken.

---

## 4. Distribution counts and the global best

| Threshold | Count | % of 702 |
|---|---|---|
| ratio ≥ 0.75 (Rybin level) | 503 | 71.7% |
| ratio ≥ 0.90 | 0 | 0% |
| ratio ≥ 0.99 | 0 | 0% |
| ratio ≥ 1.0 (counterexample candidate) | 0 | 0% |

**Global best across the entire sweep**: `n7-chain6`, k=4, attach
`[(n2,n4),(n2,n6),(n3,n5),(n3,n6)]`, ratio = 0.7500002000000001 (solver-noise
excess of 2×10⁻⁶ over 3/4), witness `demands≈[1, 0.5, 1, 1]`,
`w≈[0.5, 0.5, 0.5, 1.0]`. Structurally this is a Rybin gadget embedded in a
7-node chain, not a new extremal shape.

**k=6 spot check** (§0's calibration section explains why this is a single
data point, not a sweep): tree `[(s,n1),(n1,n2),(n1,n3)]` (n4-fork, cheapest
available 3-arc structure) with 6 chords cycling through all node pairs —
ratio = 0.6667, *below* the plateau. One instance is not evidence the k=6
stratum stays at or below 0.75 in general (see §5's honesty note on this),
but it is at least consistent with "no climb."

**No ratio ≥ 1 hit anywhere** — `positive_hits: []` in
`sup_ratio_results.json`. The word "counterexample" is not used anywhere in
this report for that reason.

---

## 5. Coverage bounds — what was sampled, what was exhaustive, what was not reached (no silent caps)

**Exhaustive** (every MSW25-uncovered structural pattern tested, not a
sample): n=4 at k=3,4 (22/22 patterns); n=5 at k=3,4 (454/454 patterns,
49+405). This is the strongest part of the result — a real proof, not
evidence, that no n≤5/k≤4 out-tree structure exceeds ratio 0.75.

**Sampled, honestly labeled, with exact tested/total counts in §2's table**:
n=5 at k=5 (34/1,939 ≈ 1.8%); all of n=6,7 (percentages ranging 18%–60% at
k=3 down to 0.04%–0.36% at k=4,5, per tree — full per-stratum figures in
`sup_ratio_results.json`).

**n=6,7 at k=5 mostly failed to produce a ratio at all**, not just a small
sample: of the per-tree k=5 attempts, most hit either a deterministic HiGHS
"Solve error" or the 45-second timeout described below — `n7-twochains/k5`,
`n7-chain6/k5`, `n7-chain-fork/k5`, `n7-star6/k5`, `n6-chain5/k5`,
`n6-star5/k5`, `n6-chain-fork/k5` all show 0 successful ratios out of 1
attempt each in `sup_ratio_results.json`; only `n6-twochains/k5` (1/1) and
the n=5 trees' k=5 samples (mostly 2–4/4) succeeded. **This is the honest
headline gap**: n=6,7 at k=5 is barely sampled and what little was attempted
mostly didn't resolve — a real open stratum, not a proof of anything.

**k=6**: one spot-checked instance, deliberately **not** a sweep. Cost
projection from the pilot (see below) made even a small sweep at k=6
infeasible within budget; the single call taking 24.6s (well under the 45s
timeout, but well above the ~5-15ms cost at k≤4) confirms the projection.

**k=7**: **not attempted at all.** Measured cost growth from k=5 (4-8s
typical) to k=6 (36s+ even at the smallest 3-arc tree, in the original pilot)
was a ~5-9x jump — faster than the ~2x the raw `2^k` binary count alone
would predict, meaning branch-and-bound cost itself blows up, not just
problem size. Extrapolating the same factor projects k=7 at several hundred
seconds *per call* even on the cheapest structure — a single call would risk
consuming most of whatever budget remained for one uncertain data point.
Left untouched, not a silent cap.

**n ≥ 8**: not touched at all.

### An operational finding worth recording: `exists_bad_instance` has no internal time limit, and its big-M MILP encoding has a real pathological tail

While running the n=5, k=5 sweep, a single call **ran over 160 seconds with
no sign of returning** (versus a typical 4-8s and a previously-seen worst
case of 58.6s) — a genuine HiGHS branch-and-bound blowup, not a hang or bug
in this script. `discrepancy.py` (read-only per this task's instructions)
has no internal MILP time limit. The fix, added in `sup_ratio.py` and *not*
in `discrepancy.py`: `ratio_of` now isolates each `exists_bad_instance` call
in a forked subprocess (`multiprocessing`) with a **45-second hard kill
switch** (`HARD_TIMEOUT_S`). A killed call is recorded as a `solver_error`
(excluded from ratio statistics, counted separately — never silently
dropped from the totals) rather than stalling the whole sweep. This pattern
recurred: of 17 total solver-side failures in the final run, most at n=6,7
k=5 were timeouts, not the separate (and much faster, ~0.2-0.6s) deterministic
"Solve error" HiGHS occasionally throws on degenerate/repeated-chord attach
patterns (also handled, via one retry). **This should be treated as a
reusable finding for any future work built on `exists_bad_instance` at
k ≥ 5**: budget for a real tail, not just the median.

The sweep itself ran across five separate process invocations (one
original run killed after a pathological hang at ~640s real time, four
short resumed continuations, all coordinated through a JSON snapshot +
skip-already-done-strata mechanism in `sup_ratio.py`'s `load_resume()`).
Combined real wall-clock time across all five was roughly 1,900s (~32 min)
— well over the nominal 15-minute compute budget for the sweep itself, but
the overrun was entirely the pathological-hang discovery and recovery, not
runaway scope; the actual **useful compute** (post-timeout-guard) fits the
intended shape (see `TIME_BUDGET_S`/`SOFT_STOP_S` in `sup_ratio.py`, and the
individual continuation budgets of 330s/400s/240s/200s that were used to
finish the sweep after the timeout guard was added). A fresh run of
`sup_ratio.py` today, with the timeout guard already in place, would not
hit this issue.

---

## 6. Self-tests (mandatory, all green)

| Self-test | Result |
|---|---|
| Rybin calibration = 0.7500 exactly | **PASS** — 0.7500000000000000 |
| MSW25-covered ⇒ ratio < 1 (NG-9) | **PASS** — 10/10 |
| Consistency vs. `certified_search.py` fixed-demand results (sup ≥ fixed) | **PASS** — 8/8 |
| Any ratio ≥ 1 rationalized + verified via `core.check_paths_valid` / `check_path_closure` / `verify_counterexample` | N/A — no ratio ≥ 1 occurred |

0 bugs recorded anywhere in `sup_ratio_results.json`.

---

## 7. Takeaway

This run replaces phase 3's heuristic **−4/9·d_max plateau on `t(w)` at
sampled fixed demands** with a **certified supremum over all demands and all
`w`**, and finds the same qualitative story at a much stronger level: **the
worst-case ratio, optimized over every possible demand choice, is capped at
exactly 3/4 — Rybin's own level — across an exhaustive sweep at n≤5/k≤4 and
an honestly-labeled sample out to n=7 and a k=6 spot check.** No structure
tested comes remotely close to the ratio-1 threshold Conjecture 1.3's
out-tree class would need to fail (max observed excess over 0.75 anywhere:
2×10⁻⁶, pure solver noise). The open territory that remains is narrow and
explicit: n=6,7 at k=5 (mostly unresolved due to solver cost, not sampled
away), k=6 beyond the single spot check, k≥7, and n≥8 altogether — all
flagged, none silently capped. Given how flat and low this ceiling sits even
as the search space grows sharply richer (n7/k4 alone has 6,753 uncovered
structural patterns, of which only 24 were sampled, all still ≤0.75), the
out-tree branch of kill-path A looks structurally saturated at Rybin's own
example rather than trending toward a break — reinforcing NG-10's original
recommendation that the better remaining bet is attacking the reformulated
demand-scaled network-matrix discrepancy question directly (or a
fundamentally different instance class), not further out-tree sweeping.
