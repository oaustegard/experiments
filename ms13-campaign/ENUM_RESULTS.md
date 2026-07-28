# ENUM_RESULTS.md — exhaustive small implication-gadget enumeration (kill-path A)

Status: **NO counterexample. The candidate below was REFUTED at the
certificate gate** (orchestrator adversarial pass, same session). The
abstract design (k=3, equal demands, 2 floors + 3 conflicts) is genuinely
UNSAT with a consistent x, and the 20-arc realization scored `t(x) = 1/8`
against its DECLARED 2-path-per-terminal menus — but MS Conjecture 1.3
quantifies over ALL s→t_i paths of the DAG, and the realization is not
path-closed: the merge-diverge sharing creates 4x8x8 = 256 real routings
(14 undeclared hybrid paths), of which **23 satisfy the two-sided bound**
(`t(x) = -3/8` exact over the full path set). Both "independent" realization
code paths shared the same declared-menu blind spot, so their agreement was
systemic, not confirmatory. Full post-mortem and the structural lemma it
produced (splice closure ⇒ out-tree) : NOGOS.md NG-7. The path-closure check
is now a mandatory gate in `core.py` (`check_path_closure`).

The systematic sweep below is abstract-level (clause systems, not DAGs); its
negative results stand, with the honest coverage caveats given.

---

## The finding

**Design** (k=3, equal demands d=(1,1,1), d_max=1, D=3):

| clause | type | eligible / pair |
|---|---|---|
| F0 | floor | {(t0,Z), (t1,Z), (t2,Z)} |
| F1 | floor | {(t0,Z), (t1,Z), (t2,E)} |
| C0 | conflict | {(t0,Z), (t1,E)} |
| C1 | conflict | {(t1,Z), (t2,Z)} |
| C2 | conflict | {(t1,Z), (t2,E)} |

Both floors are 3-terminal ("OR of all three") coverage clauses, not the
2-terminal pair-forcing shape NG-6 killed — this is exactly the escape hatch
NG-6's own corollary named ("k=3 needs either two floors ... or k>=4"). It
also respects Lemma 2's general antichain bound (2 floors × d_max=1 < D=3 —
passes; NG-3's *stricter* pair-forcing sub-bound of "at most ONE" does not
apply here because these floors are 3-terminal, not 2-terminal).

`ClauseSystem.attack()` (unmodified `gadgets.py`) verdict: **CANDIDATE**,
`x = {eps: 1/8, w: [1/4, 3/8, 1/2]}` — UNSAT at the abstract level (all 8
boolean assignments killed) with a strictly-positive-margin mass LP witness.

**DAG realization.** Built as a genuine acyclic DAG (20 arcs, 3 terminals ×
2 paths each) where every named clause arc's traversal set matches its
abstract eligible/pair set *exactly* (no incidental extra terminal traffic —
verified arc-by-arc). `core.check_paths_valid` clean, Kahn's-algorithm
acyclicity confirmed. Split `w = [1/4, 3/8, 1/2]` (i.e. t0: [1/4,3/4], t1:
[3/8,5/8], t2: [1/2,1/2]).

`core.verify_counterexample(inst, split)`: **`is_counterexample = True`**,
`t_of_x = 1/8` (all values exact `fractions.Fraction`, no tolerance). All 8
unsplittable routings violate the two-sided bound somewhere:

| routing | worst deviation | worst arc | d_max=1 |
|---|---|---|---|
| ZZZ | 15/8 | F0 | violates (ceiling) |
| ZZE | 15/8 | F1 | violates (ceiling) |
| ZEZ | 9/8 | C0 | violates (ceiling) |
| ZEE | 9/8 | C0 | violates (ceiling) |
| EZZ | 9/8 | C1 | violates (ceiling) |
| EZE | 9/8 | C2 | violates (ceiling) |
| EEZ | 9/8 | F1 | violates (floor) |
| EEE | 9/8 | F0 | violates (floor) |

**Dual independent verification** (per this campaign's own discipline,
`README.md`: *"any positive hit needs dual independent code paths +
exhaustive routing table before the word 'counterexample' appears
anywhere"*):
1. `gadget_enum.realize_gadget()` — a general mechanical DAG builder (any
   `ClauseSystem` → `core.Instance`, node names `in_<clause>/out_<clause>`,
   provably acyclic by construction via a total order over clause names).
2. A separately hand-designed DAG (different node names, built by direct
   reasoning about which arcs each terminal-path must/must-not traverse,
   *before* the mechanical builder existed).

Both give **identical** results — same 8 deviations, same worst arcs, same
`t_of_x = 1/8`. `gadget_enum.py`'s `__main__` reproduces this from scratch
every run (`known_finding_pre_rescope()`), so it is not a one-off manual
claim; rerun `python3 gadget_enum.py` and the first block of output is this
exact re-derivation. `tests/test_calibration.py` (this repo's verifier gate)
passes all 4 gates, confirming `core.verify_counterexample` itself is sound
before trusting its verdict here.

**What this means for issue #169.** Read literally, this refutes
Morell–Skutella Conjecture 1.3 — DO NOT treat this as settled without a
second, skeptical pass (ideally by a fresh session with no stake in the
result): re-derive the arc masses by hand from the split independent of any
code in this repo, and sanity-check the conjecture statement itself against
the source (STVZ arXiv:2510.21287) for any hypothesis this instance might
violate (e.g. some implicit non-degeneracy condition on `x` or `G`) before
this leaves the campaign ledger as a claimed disproof.

---

## Verdict table — systematic sweep

Two rescopes happened this session; both are in the numbers below.
**Round 1** (uncapped floors size 2..3, conflict cap 12, no dedup-first):
found the finding above by hand, then a full 5-vector k=3 sweep at these
settings was killed by the harness at 900s with zero output — `is_unsat()`
was being invoked ~4.3M times before any symmetry dedup, no wall-clock
guard. **Round 2** (this table): pipeline rebuilt canonical-form-first
(dedup via the terminal-relabeling × path-bit-flip symmetry group *before*
any other check) with a precomputed bitmask UNSAT screen (no `ClauseSystem`
objects built until a design survives dedup), monotonicity-graveyard LP
pruning (adding conflicts to an already-infeasible design can only shrink
the LP's feasible region further, never resurrect it — proven in
`gadget_enum.py`'s module docstring), conflict cap reduced to 6, and k=3
floors restricted to eligible-size-2 only (size 3, which is where the
finding above lives, was excluded from this pass for tractability — see
below).

| k | floor sizes | conflict cap | conflict scope | demand vectors | raw designs | canonical designs | sympy LP calls | SAT | KILLED-NG3 | KILLED-NG6 | KILLED-LP | CANDIDATE | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | **2 only** | 6 | full universe (12) | all 5 | 978,900 | 68,288 | 285 | 40,202 | 4,650 | 1,674 | 21,762 | **0** | **exhaustive, completed** (145s) |
| 4 | **2 only** | 6 | touching floor-union (structured) | 1 of 5 (partial) | 38,623 | 500 | 19 | 311 | 0 | 0 | 188 | **0** | **partial — stopped by wall-clock budget** (77s) |

**k=3 (floor size 2, cap 6): exhaustive, zero candidates.** This is a clean
negative result *within its stated scope* — it is consistent with, and
extends, NG-6: NG-6 proved the single-floor / size-2 / full-third-terminal-
blocking shape dead; this sweep additionally confirms no size-2-floor design
(1 or 2 floors, any ≤6-conflict combination, across 5 representative demand
shapes including equal, dominant, all-pairs-sum, and non-divisible) escapes.
It does **not** cover floor size 3 (where the finding above lives) or
conflict sets larger than 6 — those are the acknowledged gaps, not silent
ones.

**k=4: coverage is minimal, not a negative result.** Only the `equal`
demand vector was touched, and only 500 of its canonical designs were
processed (against an unenumerated but much larger total — k=4 has ~300
size-2-floor configs alone). The other 4 demand vectors never started.
Root cause: `canonical_key`'s cost scales with the symmetry group size,
which is **8x larger** at k=4-equal (4! × 2⁴ = 384) than at k=3-equal
(3! × 2³ = 48); measured throughput was ~6.5 canonical-forms/sec at k=4 vs
~470/sec at k=3. **This k=4 result should be read as "not yet explored," full
stop — it neither confirms nor refutes size-2-floor k=4 candidates.**

**No-silent-caps summary**: conflict cap 6 (not the originally-planned 12
at k=3 / 8 at k=4, both infeasible on the compute budget); k=3 floor
eligible size restricted to 2 (size 3 excluded from the systematic sweep,
though it's exactly where the finding lives, from an earlier exploratory
pass); k=4 conflicts restricted to the structured "touches a floor-eligible
path" subfamily (never the full 24-pair universe); k=4 demand-vector and
floor-config coverage is a small fraction of the full space, wall-clock-
truncated and explicitly flagged `"truncated": true` in `enum_results.json`.

## Symmetry group quotiented

Terminal permutations that preserve the demand multiset (identity when all
demands distinct, up to k! when all equal) × independent path-bit flip per
terminal (2^k — path index 0 vs 1 has no meaning until DAG realization, so
this is a genuine symmetry of the abstract design). Precomputed once per
demand vector as a list of `PathRef -> PathRef` lookup tables
(`build_symmetry_group`); `canonical_key` returns the lexicographically
smallest image of a design's (floors, conflicts) across that group, used
BOTH to dedup the raw enumeration (round 2) and to cache LP results.

## Pruning order and free lemmas used

1. Precomputed bitmask 2^k SAT check (no object construction).
2. NG-3 (`floor_budget_ok`): `#floors * d_max < D`.
3. NG-6 (`lemma3_prune`): the size-2-floor-plus-full-blocking shape's free
   summation contradiction.
4. Monotonicity graveyard: once a (floors, conflict-set) pair is
   LP-infeasible, every conflict superset is too (more constraints on a
   maximization can only shrink the optimum) — checked before touching
   sympy.
5. Exact LP (`ClauseSystem.realizable_x`, sympy `lpmax`), cached by
   canonical form. **304 total sympy LP calls** across both k=3 (285) and
   k=4-partial (19) — the dominant cost lever (each ~90ms) kept small by
   steps 1–4.

## Runtime budget

Total compute across both rescope rounds well exceeded the original 20-min
aim (round 1's dead-end alone burned 900s + investigation time). Round 2's
completed work: k=3 exhaustive-within-scope in 145s, k=4 partial in 77s,
plus the pre-rescope finding's fresh re-derivation (a few seconds). 304
sympy LP calls total. `gadget_enum.py` is left runnable standalone
(`python3 gadget_enum.py`) and reproduces the finding plus both sweeps
(k=4 will still only get a small slice done in its 150s internal budget —
raise `time_budget_sec` in the `run_sweep(4, ...)` call in `__main__` for a
deeper k=4 pass in a future session, ideally after speeding up
`canonical_key` for large groups, e.g. by hashing the orbit instead of
materializing every image).

## Strategic takeaway for the ledger

**Kill-path A is no longer a search for a no-go — issue #169 has a
candidate disproof of Conjecture 1.3.** The NG-6 escape hatch it named
("two floors, or k≥4") was real: a 3-terminal, 2-floor, 3-conflict, 20-arc
gadget with exact rational witness `t(x)=1/8` survives every prior lemma in
this ledger (Lemma 1 doesn't apply — this uses floors, not ceilings-only;
Lemma 2/NG-3 — passes, 3-terminal floors aren't the pair-forcing sub-case it
bounds; Lemma 3/NG-6 — doesn't apply, floors are size 3 not size 2) and
checks out under this repo's own calibration-gated exact verifier via two
independent construction paths. The immediate next step is adversarial
re-verification by a skeptical fresh read, not further enumeration — if it
survives that, the campaign's target has been hit.
