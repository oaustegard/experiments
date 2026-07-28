# CERTIFIED_RESULTS.md — phase-4 certified out-tree search (supersedes phase 3's heuristic)

**Status: NO counterexample found. Certified negative, not a heuristic one.**
Total run: 150.4s of the 900s (15 min) budget — 0 bugs, 0 positive hits, every
self-test green. Search code: `certified_search.py`. Machine-readable
results: `certified_results.json`.

---

## 0. What "certified" means here

Phase 3 (`outtree_search.py`) maximized `t(w)` per instance by grid+polish —
a heuristic **lower bound**. `discrepancy.exists_bad_w(inst)` (built and
calibrated in the session that produced NOGOS.md NG-11, GATE 5) **decides**
the same question exactly via MILP: does ANY `w ∈ [0,1]^k` make every
`z`-rounding violate? It returns `proved_no_bad_w: True` when infeasibility
is certified, not just "not found." Measured cost in this run: **2-10ms at
k≤5, 3-800ms at k=6** (occasional solver-tail outliers, never a hang).
Because it's this cheap, this run could afford to certify **every** uncovered
structural pattern at n≤5 (phase 3 sampled ≤15 patterns/combo), not just a
capped sample.

---

## 1. Strata table

### Stage 1 — n ≤ 5 (exact re-enumeration of phase 3's space)

| | count |
|---|---|
| structural patterns (deduped) | 16,453 — **identical to phase 3's ledger** |
| covered (MSW25/NG-9, provably dead) | 14,038 (85.3%) |
| uncovered (live) | 2,415 (14.7%) |
| uncovered patterns certified via `exists_bad_w` | **2,415 / 2,415 (100%)** |
| demand vectors per pattern | 3 (see §2) |
| total `exists_bad_w` decisions (live) | 7,245 |
| **decided infeasible (`proved_no_bad_w`)** | **7,245 / 7,245 (100%)** |

Every uncovered pattern at n≤5 is now certified, not sampled — a strictly
stronger statement than OUTTREE_RESULTS.md's "no positive t(w) in the sampled
subset."

### Stage 2 — n = 6, 7 (new)

8 tree shapes (chain-shaped + fork/star/twochains variants per n), k = 3..6,
32 (tree,k) combos.

| n | raw combos (Σ) | patterns classified (Σ) | covered | uncovered | uncovered % |
|---|---|---|---|---|---|
| 6 | 216,512 | 58,071 | 40,947 | 17,124 | 29.5% |
| 7 | 1,183,028 | 61,442 | 42,153 | 19,289 | 31.4% |

By k (both n, aggregated):

| k | covered | uncovered | uncovered % |
|---|---|---|---|
| 3 | 6,843 | 555 | 7.5% |
| 4 | 32,949 | 8,254 | 20.0% |
| 5 | 32,086 | 14,668 | 31.4% |
| 6 | 11,222 | 12,936 | 53.6% |

**The uncovered fraction keeps climbing with both n and k** — continuing
phase 3's trend (n=2,3: 0%; n=4: ~4%; n=5: ~15%; n=6/7: ~30%, k=6 alone:
~54% within the sampled set). Larger, richer trees create K4 minors more
easily, exactly as NG-9's structural argument predicts.

Enumeration was **exhaustive** for 20/32 combos (raw combo count ≤ 20,000);
**sampled** (4,000 random raw attach tuples, honestly labeled) for the other
12, all at k=5 or k=6 where the raw with-replacement space exceeds 50,000
(up to 1,183,028 raw combos for n7/k6 — brute enumeration there is not
"screening," it's a different-scale problem).

Live certification: 3,549 `exists_bad_w` decisions across 32 combos, capped
per combo (60 patterns for k≤5, 15 for k=6, × 2-3 demand vectors) for time
budget — **9 combos got their full uncovered set tested, 23 were a capped
sample** (explicit list in `certified_results.json`). **All 3,549 decided
infeasible.**

### Grand total

**10,794 `exists_bad_w` live decisions (stage 1: 7,245; stage 2: 3,549),
plus 639 self-test/cross-check `exists_bad_w` calls (stage 1: 236 NG-9 +
74 Lemma 6 + 6 permutation-invariance + 35 phase-3-argmax-recheck = 351;
stage 2: 192 NG-9 + 96 Lemma 6 = 288) — 11,433 total MILP decisions, plus a
separate, non-MILP 20-check Lemma-7 cycle-shift verification (pure exact
arithmetic, no solver). 0 bugs, 0 hits, 0 inconclusive results**, in 150.4s.

---

## 2. Heuristic vs. certified — the key comparison

Two independent comparisons, both agreeing:

1. **Phase 3's own recorded argmax instances, re-decided.** For every
   (tree, k) combo where phase 3 recorded a `max_t_found_uncovered` (35
   instances, spanning every n≤5 tree/k with a nonempty uncovered set — this
   includes the global-best `n5-chain-fork` and `n5-chain2-2leaf` instances
   at `t* = -4/3`), `exists_bad_w` was run on the *exact* (attach, demands)
   phase 3 reported. **All 35 agree with phase 3's negative verdict** —
   every one comes back `feasible: False, proved_no_bad_w: True`. No
   disagreement anywhere.

2. **Every uncovered structural pattern at n≤5, certified against 3 demand
   vectors each (7,245 decisions)** — not just phase 3's argmax points, the
   *entire* uncovered set phase 3 only partially sampled (capped at 15
   patterns/combo, 4 random demand/permutation trials each — 1,312
   instance-searches total in phase 3, vs. 7,245 exact decisions here).
   **100% proved infeasible.**

**Verdict: zero disagreement.** The certified pass does not just confirm
phase 3's specific findings — it closes the *entire* uncovered structural
space phase 3 could only sample. This is meaningfully stronger than "the
heuristic search didn't find anything": it is now a proof that at n≤5, for
every uncovered structural pattern and every demand vector tested, no
counterexample of this shape exists.

**Demand vectors used** (kept identical to phase 3's named set where
possible, for direct comparability): k=3: `(15,10,15),(7,5,3),(5,5,6)`; k=4:
`(9,7,6,4),(11,9,6,2),(13,8,5,3)`; k=5: `(11,9,7,5,3),(13,11,8,6,2),
(9,7,6,5,4)` — the first 3 of phase 3's `build_demand_vectors()` unequal
list (dropped the 2 random ones per k to control cost; see §5 for the
permutation-invariance argument that makes this a non-issue on the
*pattern* axis, though it does leave demand-value diversity as a genuine,
stated limitation — see §6).

**Permutation invariance, verified.** `exists_bad_w` searches over the
*full* box `w ∈ [0,1]^k` for *all* z simultaneously, so relabeling which
demand value sits at which attach slot cannot change the reachable set of
`t(w)` values (apply the inverse relabeling to `w`). Checked empirically on
3 instances (identity vs. a random permutation of both `attach` and
`demands`): **all agreed exactly** (`feasible_a == feasible_b` in every
case). This is why the live search below tests each (pattern, demand
*multiset*) once rather than phase 3's k! permutation trials — a genuine
simplification specific to the certified decision, not available to the
heuristic search (whose finite grid *is* permutation-sensitive in practice,
which is presumably why phase 3 ran permutation trials at all).

---

## 3. Lemma 7 / Corollary 7.1 — how it was used (and how it wasn't)

**Used as a verification self-test**, extended to n=6,7: whenever a randomly
sampled attach pattern's *full* chord multigraph (all k chords, not just a
fractional subset) contains a cycle, `discrepancy.find_chord_cycle` +
`cycle_shift_preserves_loads` confirm the signed cycle combination vanishes
on every tree arc — Lemma 7's core claim. **20/20 checked, 20/20 passed** on
n=6,7 instances (NG-11 verified it on 233 n≤5 cycles; this extends the check
to the new tree sizes).

**Used qualitatively** to choose which n=6,7 shapes to add: Corollary 7.1
says a counterexample needs a K4 minor (≥4 branch-forming tree nodes,
NG-9), so shapes with more branching opportunity (fork, star, twochains — not
just pure chains) were included alongside the chain-shaped baseline, on the
theory that they reach K4-minor-worthy configurations with fewer chords.
Confirmed post-hoc by the enumeration counts: fork/twochains shapes have
uncovered fractions comparable to chains at the same k (see
`certified_results.json`'s `stage2_n67.enumeration_counts`).

**NOT used as a structural-enumeration pruning device**, deliberately. The
lemma's precondition is about which chords are *fractional at an LP-optimal
z-relaxation for a fixed instance* — a fact about a specific instance's
critical point, not a license to skip generating certain *structural attach
patterns* in the outer combinatorial search. Turning "≤n-1 chords fractional
at criticality" into "don't bother enumerating patterns with >n-1 distinct
chords" would need an argument that the *skipped* patterns are covered by
what the *kept* ones already test — not obviously true, and not something
this run's budget allowed proving carefully. Given `exists_bad_w` is cheap
and exact regardless of which patterns get tested, the budget went to more
MILP calls instead of a pruning heuristic that might have silently narrowed
the search. This is an explicit, considered choice, not an oversight.

---

## 4. Self-tests (mandatory, all green)

| Self-test | n≤5 (stage 1) | n=6,7 (stage 2) |
|---|---|---|
| NG-9 (MSW25-covered ⇒ `exists_bad_w` infeasible) | 236/236 | 192/192 |
| Lemma 6 (equal demands ⇒ `exists_bad_w` infeasible) | 74/74 | 96/96 |
| Lemma 7 (cycle-shift preserves loads) | — (NG-11 already covers n≤5) | 20/20 |
| Permutation invariance (new check, not mandated but load-bearing for §2) | 3/3 | — |

Zero assertion failures anywhere. Combined with the live-search 100% pass
rate, this run found no evidence of an encoding bug (the two that were caught
during `exists_bad_w`'s original construction, per NG-11, are not
reproduced here).

---

## 5. Cost projection vs. actual

Stated before running: pilot measurements gave ~2-15ms/call at k≤5 and
~15-120ms/call at k=6 (with occasional 300-800ms outliers), so a projection
of "a few minutes for stage 1, a few more for stage 2, comfortably inside
15 minutes" was made and knobs (3 demand vectors/pattern at n≤5, capped
60/15 patterns per combo at n=6,7) were chosen for that target.

**Actual: 150.4s total** (79.7s stage 1, 70.7s stage 2) — about **17% of the
900s budget**. The projection was conservative; real per-call costs in the
full run were faster than the worst-case pilot numbers (median MILP solves
dominate; the rare multi-hundred-ms outliers didn't compound). This leaves
**~750s of unused budget** — see §6 for what a follow-up pass with that
margin could add.

---

## 6. Coverage bounds — what was NOT reached (no silent caps)

- **n ≤ 5**: every uncovered structural pattern was certified (100%), but
  only against **3 of phase 3's demand vectors per k** (dropped 2 random
  ones/k for cost) and **one demand-to-slot assignment per pattern**
  (justified exactly by the permutation-invariance argument in §2, not a
  coverage gap on that axis). The **pattern axis is now exhaustive**; the
  **demand-value axis is still a finite sample** (3 named vectors, not a
  continuum or an exhaustive rational sweep) — a genuine, if minor, residual
  gap given `t(w)`'s dependence on the exact demand *ratios*, not just the
  attach structure.
- **n = 6, 7**: structural enumeration was exhaustive for 20/32 combos and a
  4,000-sample (of up to 1.18M raw) for the other 12 (all k=5,6) — labeled
  per-combo in `certified_results.json`. Live certification then sampled
  *within* the uncovered set for 23/32 combos (capped at 60 patterns for
  k≤5, 15 for k=6, out of uncovered sets as large as 3,960) — only 9/32
  combos got their full uncovered set live-tested. This is the honest
  headline limitation of stage 2: **most of the n=6,7 live territory is
  sampled, not exhaustively decided.**
- **n ≥ 8, k ≥ 7**: not touched at all.
- **Automorphism dedup on samples**: for the 12 sampled (non-exhaustive)
  enumeration combos, dedup by tree automorphism is only within the sample,
  not a claim of full orbit coverage.
- **Budget headroom unused**: 150.4s of 900s. A direct next step (not run
  here, to keep this pass's scope matched to its stated plan) would raise
  `CAP_LIVE_K345`/`CAP_LIVE_K6` and `SAMPLE_RAW`, and/or add the 2 dropped
  random demand vectors per k back in, using the ~750s margin — likely
  pushing several of the 23 capped n=6,7 combos to full coverage.

**Interpretation.** The *n≤5 negative is now essentially airtight* within
this instance class (exhaustive on patterns, a small finite demand sample).
The *n=6,7 negative is suggestive but explicitly not exhaustive* — same
epistemic status phase 3's n≤5 sweep had before this pass, one level up in
n. The uncovered fraction climbing with n and k (7.5%→20%→31%→54% by k, 0%→
~4%→~15%→~30% by n) is the more actionable structural finding: **the live
territory is growing, not shrinking**, so n=8+ is where the next real
budget should go if this branch continues.

---

## 7. Takeaway

Phase 4 certifies, rather than merely fails to falsify, phase 3's negative
result at n≤5: every one of 2,415 uncovered structural patterns is now
*proved* (not sampled) to admit no bad `w` under 3 representative unequal
demand vectors, and phase 3's own reported argmax instances all
independently re-verify as infeasible — **zero disagreement between the
heuristic and the certified pass**. The extension to n=6,7 (new territory,
not swept by phase 3) finds the same pattern — 3,549/3,549 certified
infeasible — but is honestly a sample, not a proof, given the combinatorial
scale (up to 1.18M raw attach patterns at n=7/k=6). The one clear trend
worth acting on: **uncovered (MSW25-live) fraction keeps rising with both
n and k** (0% at n≤3, ~15% at n=5, ~30% at n=6-7, ~54% within k=6 samples) —
if kill-path A continues down the out-tree branch, n=8+ chain/fork shapes
are the next frontier, not a re-sweep of what's already certified here.
NG-10's phase-3 recommendation to pivot to the reformulated
demand-scaled-network-matrix discrepancy question directly (rather than
sweeping more instances) still stands as the better long-term bet — this
run strengthens the "no counterexample found yet" evidence without closing
that question.
