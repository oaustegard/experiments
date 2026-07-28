# Rule search: does a deterministic rounding rule certify R <= 3/4?

Task C only, per the coordinator's rescoped remit (Task A: exhaustive
enumeration, and Task B: exact B&B proof R<=3/4 on both maximal types, done
elsewhere — see the coordinator's own `K3_PROOF.md`). This note covers the
constructive-rule search: `run_rule_search.py`, using `k3_proof.py`'s row-set
enumeration (loaded from `/tmp/k3_enum_progress.json`, 30 types, reused not
recomputed) and four new rule implementations added to `k3_proof.py`
(`rule_a_decreasing_demand` ... `rule_d_small_side_first`, `sweep_rule`).
Exact `fractions.Fraction` throughout — no floats anywhere in this file's
numbers.

## Rules tested

- **(a) decreasing demand, greedy min-max**: process chords in decreasing
  `d_i = p_i+q_i`; at each step choose `delta_i in {q_i, -p_i}` to minimize
  the resulting `max_a |partial load|` given prior choices.
- **(b) decreasing max(p_i,q_i), greedy min-max**: same greedy step as (a),
  different processing order.
- **(c) oppose current aggregate load**: decreasing-`d_i` order; at chord
  `i`, compute `score = sum_a C[a,i]*partial[a]` over rows touching `i`; if
  `score > 0` (increasing `delta_i` would push those rows further from 0),
  take `delta_i = -p_i`; else take `delta_i = +q_i`.
- **(d) small-side-first**: decreasing-`d_i` order; prefer the option with
  smaller `|delta_i|` unless it strictly worsens the resulting max vs the
  other option (differs from (a) only in tie-breaking, in principle).

## Method

`sweep_rule` runs each rule on every one of the 30 enumerated row-set types,
against 500 random exact-rational `(p,q)` points (denominators 2..12, drawn
uniformly, `q_i` clipped to keep `p_i+q_i<=1`) plus **mandatory known
extremal witnesses**: the Rybin type's `d=(1/2,1,1)`, `w=1/2` point
(`p=q=(1/4,1/2,1/2)`, independently confirmed via `exact_min_over_z` to hit
exactly 3/4 before any rule ran — sanity gate in `run_rule_search.py`), and
each maximal type's own exact witness (from `k3_proof.py`'s
`find_exact_witness`). **Caveat on witness reuse**: a witness `(p,q)` is only
meaningful paired with the *exact* row tuple it was derived against —
canonicalization (column permutation + sign flip) scrambles which `p_i`
belongs to which column. The first version of this script hand-copied the
Rybin witness from its raw `[(1,1,1),(1,1,0),(1,0,1)]` realization and paired
it with the canonical `((0,1,-1),(1,-1,0),(1,-1,1))` row tuple — caught
immediately by the sanity assert (`exact_min_over_z` returned `1/4`, not
`3/4`) before any rule result was trusted. Fixed by deriving every witness
in-place against its own stored canonical form. Full run: 4 rules x 30 types
x ~503 points = ~60,000 exact-rational rule evaluations, 7.5s total.

## Result: all four rules FAIL

| rule | types failed (of 30) | worst value seen | tightest failure |
|---|---|---|---|
| (a) decreasing demand | 17 | **1** (= d_max, not just >3/4) | 7/9 |
| (b) decreasing max(p,q) | 11 | **1** | 4/5 |
| (c) oppose current load | **30** (all) | 3/2 (exceeds d_max by 50%) | — |
| (d) small-side-first | 17 | **1** | 7/9 |

None of the four candidate rules certifies `R<=3/4`. (c) is not close —
it fails on every single type and can exceed even the un-strengthened
`d_max=1` bound (Conjecture 1.3's own threshold) by a wide margin, so
"oppose the current load" is actively a bad heuristic here, not just an
insufficiently strong one.

**The dramatic common failure point** for (a), (b), and (d): row-set
`{(0,0,1),(0,1,-1),(1,-1,1),(1,0,1)}` at `p=q=(1/2,1/2,1/2)` (i.e. all
demands equal to 1, `w=1/2` uniformly). All three rules pick `z=(1,1,1)`,
giving `max_a|load|=1` — but the TRUE optimum over all 8 roundings at this
exact point is `1/2`, attained at `z=(0,1,1)` (verified via
`exact_min_over_z`, the brute-force independent check). This is not a
narrow miss: greedy locks in a locally-good first choice (on the
highest-demand chord) that turns out globally wrong, and single-pass
processing has no way to revisit it. `d`'s only difference from `a` is
tie-breaking, and it makes no difference at this witness — same failure,
same z.

Rule (a) — decreasing-demand greedy — is closest to working (17/30 fail,
tightest violation 7/9, only slightly over 3/4) but still concretely
false as a universal certificate: rule (b)'s reordering by `max(p_i,q_i)`
instead of `d_i` cuts the failure count from 17 to 11 without eliminating
it, which itself is informative — the choice of processing order matters
a lot but no single fixed order examined here removes the failure mode.

## Read on generalizability

**Negative result, and it is informative.** All four single-pass
greedy/local rules fail, several dramatically (reaching the RAW `d_max`
bound, not just missing 3/4 by a little). The common failure shape —
locking in the highest-demand chord's sign first, then discovering two
steps later that the OTHER two chords' interaction with that early choice
was the actual problem — says the extremal configurations are **genuinely
simultaneous**: no single scalar priority order (by demand, by
`max(p,q)`, by "current load") separates the chords into a safe
processing sequence, because which chord should go "first" depends on
the *joint* structure of all three rows, not a property of one chord
in isolation. This matches the qualitative shape of the maximal-type
witnesses themselves (both hit exactly 3/4 at points with **all three
demands equal to 1 and at least one `w_i` off-center** — maximal mutual
frustration, not something a scalar ranking sees coming).

Given this, **the k=3 case analysis (exact B&B over the 2 maximal types)
currently looks like the only working route for k=3**, and none of these
four rules is a promising building block for a constructive proof at
general k — a rule that fails on 3-chord instances this small isn't going
to do better once there are more chords to get the ordering wrong on. That
said, the search here covered exactly four hand-picked rules out of a much
larger space (e.g. two-pass rules, rules with backtracking/local-search
repair after an initial greedy pass, or LP-relaxation-guided rounding) —
this result rules out the four *simplest* candidates, not the entire
concept of a constructive proof.

## Files

- `k3_proof.py`: added `rule_a_decreasing_demand`, `rule_b_decreasing_maxpq`,
  `rule_c_oppose_current_load`, `rule_d_small_side_first`, `_rule_engine`,
  `_trial_max`, `sweep_rule`, `_RULES` (alongside the pre-existing
  `greedy_decreasing_demand`/`test_rounding_rule`, which is rule (a) under a
  different name from an earlier pass of this session — kept for backward
  compatibility with `run_full`, superseded by `rule_a_decreasing_demand` +
  `sweep_rule` for this report).
- `run_rule_search.py`: this report's driver (loads the 30-type list,
  applies the witness-reuse sanity gate, runs all 4 rules, writes
  `k3_rule_search_results.json`).
- `k3_rule_search_results.json`: full per-type, per-rule results (worst
  value, worst `(p,q)`, pass/fail) for all 30 types x 4 rules.
