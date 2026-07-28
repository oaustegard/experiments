# WORKLOG — ms13-campaign (issue #169)

Multi-session campaign state, per the iterating pattern. Newest session first.
Read this + NOGOS.md before doing anything.

## Naming: which "Conjecture 1.3"?

The literature numbers collide; the campaign uses **STVZ numbering**
(arXiv:2510.21287) throughout:

| Statement | STVZ | TVZ planar (2308.02651) | Morell–Skutella original |
|---|---|---|---|
| Goemans: cost + one-sided d_max | Conj 1.2 | Conj 1.3 | — ("conjecture of Goemans") |
| MS no-cost, two-sided d_max, acyclic — **our target** | **Conj 1.3** | Conj 1.5 | Conjecture 1 |
| MS cost, two-sided d_max, acyclic | Conj 1.4 | Conj 1.4 | Conjecture 2 |
| cost + two-sided 2·d_max, acyclic | Conj 1.5 | — | — |

Never cite "Conjecture 1.3" without this table in reach.

## Session 2026-07-24 (phase 5 — proving Conjecture 12.1 at k=3)

Scope decision (Oskar, explicit): **do step 1 only, stash the rest.**

**DONE — Lemma 8, the lower bound of Conjecture 12.1 at k=3 (NG-13).**
R depends on a structure only via its row-set (mod row/column sign flips and
column permutation) and is monotone in that set, so k=3 is a finite case
analysis: 25 types, R quantized to {1/2, 2/3, 3/4}. The minimal type attaining
3/4 has three rows — the Rybin type — and a two-case hand argument gives
`min over roundings = min(t/2 + 1/2, 1 − t/2)`, maximized at t = 1/2 → **3/4,
attained**. Verified exactly at four values of t, and the extremal point
coincides with the witness NG-12's MILP found independently.

**IN PROGRESS — the upper bound `R ≤ 3/4`** for the maximal types, as an
exact-rational disjunctive branch-and-bound (does the union of the 8 good
polytopes cover the normalized box?), plus a rigorous completeness argument
for the 25-type list and a search for a certifying rounding rule (which would
generalize past k=3). See `K3_PROOF.md`.

### STASHED (deliberately not done this session — not dropped)

1. **Exactness of the R = 3/4 *ceiling*** beyond k=3. NG-12's sweep is
   float (HiGHS). Note this is genuinely harder than "rerun with Fractions":
   certifying `R ≤ 3/4` for a structure is a universally-quantified claim over
   a polytope, needing an exact rational MILP, exact LP-duality/CEGAR
   certificates per branch, or — cleanest — the proof itself. A general proof
   would make the float computation moot rather than requiring its redo.
2. **The k ≥ 5 branch-and-bound pathology** in `exists_bad_instance` (no MILP
   time limit; >160s hangs), and the n ≥ 8 / k ≥ 7 strata it blocks. Fix =
   time limit + warm start, or a CEGAR reformulation. First task for anyone
   extending the sweep, but of no use to the proof track.

## Session 2026-07-24 (phase 4 — the 3/4 plateau, post PR #173)

**Headline: the campaign now has a sharp constant, not a null result.**
`R(structure)` = sup over ALL demands and w of the min-max deviation ratio.
Measured R = **3/4 exactly** across 702 certified decisions (n = 4..7,
k = 3..6); nothing exceeded it, 71.7% attained it, an independent adversarial
batch confirmed the cap. Extremal witness is exact and hand-checkable: the
Rybin structure with demands (1, 1/2, 1) and w = (1/2,1/2,1/2) → 3/4.
→ **Conjecture 12.1**: Conjecture 1.3 holds on the whole 2-path grammar with
constant 3/4, attained. Hence any 1.3 counterexample needs ≥ 3 paths per
terminal (Lemma 4's open edge).

**Machinery (each strictly stronger than the last):**
- **Lemma 7 (NG-11)**: cycles among fractional chords shift free of any load
  change ⇒ WLOG the fractional chords form a forest; Cor 7.1 caps fractional
  chords at n−1 regardless of k, so the minimal critical shape is 3 chords on
  4 nodes — the Rybin shape.
- **Path trees ⇒ interval matrices** (verified): the open question in its
  cleanest form is interval discrepancy with scaled columns; ≥3 mutually
  crossing intervals with unequal demands is the only live case.
- **`exists_bad_w`** (certified per-instance decision, MILP over w) replaced
  phase 3's grid heuristic: **zero disagreement**, and it closed all 2,415
  uncovered n ≤ 5 patterns exactly rather than by sampling.
- **`exists_bad_instance`** closed the demand axis: substituting
  p_i = d_i w_i, q_i = d_i(1−w_i) linearizes the system, d_max normalizes to
  1, and one MILP decides a whole STRUCTURE over all demands — with
  threshold 0 the objective *is* R.
- **GATE 5** is two-sided by design. My first MILP encoding had inverted
  big-M polarity and a flipped ε sign; both would have made it report "no
  counterexample" everywhere, and a one-sided gate would have passed them.

**Known defect (recorded, not hidden):** `exists_bad_instance` has no MILP
time limit and hits a >160s branch-and-bound pathology at k ≥ 5; the sweep
wraps it in a hard-killed subprocess and records kills as `solver_error`.
Fixing that (time limit + warm start, or a CEGAR formulation) is the first
task for anyone extending to k ≥ 6.

**Next session (phase 5):** prove or refute **Conjecture 12.1**. It is a much
better target than 1.3 itself — falsifiable by a single structure with
R > 3/4, provable by an argument that need only reach 3/4. Attack it in the
interval form (3 crossing intervals, unequal demands) where the extremal
witness is smallest. Secondary: fix the k ≥ 5 solver pathology, then extend
to n ≥ 8 / k ≥ 7 to test whether the 3/4 ceiling survives more freedom.

## Session 2026-07-24 (phase 3 — out-tree normal form, post PR #172)

**Theory (the session's product):**
- **Lemma 5 (NG-8)**: Conjecture 1.3 on path-closed 2-path instances *is*
  signed discrepancy over the tree's **network matrix** C (totally
  unimodular): `f^P(a) − x(a) = Σ_i C[a,i] d_i (z_i − w_i)`. Validated
  against the Rybin anchors through two independent code paths. Terminal-arc
  caveat found by cross-check: they can never violate but do enter the max
  (−11 vs true −9), so the predicate is tree-arc-only while the value needs
  both; `outtree.py` exposes both.
- **Lemma 6 (NG-8)**: equal demands ⇒ no counterexample in this class
  (Hoffman–Kruskal on [C;I]); 24k-eval stress test. Kills NG-7's candidate a
  second, structural way and confines the hunt to unequal demands — exactly
  where the demand scaling breaks TU (C is TU, C·diag(d) is not).
- **NG-9 — the sharpest prune yet.** MSW25 = Majthoub Almoghrabi/Skutella/
  Warode arXiv:2412.05182 Thm 2 proves 1.3 at the **tight** ±d_max bound for
  multiflows on series-parallel digraphs (arbitrary commodity sources/sinks).
  Derived criterion: suppressing terminal sinks gives tree + one chord per
  terminal, so an instance is **MSW25-covered iff that graph is K4-minor-free**.
  Rybin's chord graph is exactly K4 — the Goemans-killer sits precisely
  outside the proved class. Covered instances are now a second self-test.
- **NG-10**: 16,453 deduped structural patterns swept; 85.3% MSW25-covered,
  n ≤ 3 dead by vertex count alone (K4 minor needs 4 branch sets); live set
  plateaus at t\*/d_max = −4/9, *further* from the bound than random layered
  DAGs. Explicit coverage bounds recorded (n ≥ 6, k > 5 unswept; ~4% sampling
  at densest combos; grid-not-LP optima).

**Literature status of the remaining class**: no tree/arborescence theorem
exists (STVZ §1.1 lists only divisible demands, planar 2·d_max,
series-parallel); Doerr's TU linear-discrepancy bound (Combinatorica 2004,
≤ 1 − 1/(n+1)) covers only **unit** boxes = our Lemma 6. The demand-scaled
network-matrix question has neither theorem nor counterexample. Genuinely open.

**Next session (phase 4) — recommended pivot:** attack the reformulated
question directly rather than sweeping more instances. *Is the linear
discrepancy of a network matrix with demand-scaled columns bounded by
max_i d_i?* One clean statement, subsumes every out-tree instance at once,
resolvable either way as a real result: a proof closes the whole 2-path
grammar and forces counterexamples to ≥3 paths per terminal (Lemma 4's scope
boundary); a counterexample to it is a counterexample to 1.3. Secondary:
certified per-instance MILP optimizer for n = 6,7 chains.

## Session 2026-07-24 (phase 2 — kill-path A assault, same day, post PR #171)

**Path B: PIVOTED.** NG-4's ladder verdict stands (abstract conflict
mechanism self-limits at 7/6, far under the needed 2); no further odd-hole /
clique rungs this campaign unless a new mechanism idea appears. NG-5 closed
free of charge: the PR #167 family at b=1, r=q=5/12 is the ladder's C3 point
(β* = 7/6 exact), voiding the convention-tainted "sup = 1" finding.

**Path A theory (this session's real product):**
- NG-6 / Lemma 3 (blocking cost): fully blocking a 2-path terminal against a
  fixed path costs a demand-only split bound; kills the minimal k=3
  single-floor gadget for every demand vector.
- NG-7 / Lemma 4 (splice closure): a candidate that passed abstract UNSAT +
  mass-LP and scored t(x)=1/8 on its declared menus was REFUTED at the
  certificate gate — the realization wasn't path-closed (256 real routings,
  23 satisfying). Post-mortem yields the structure theorem: path-closed
  2-path instances are out-trees with in-degree-≤2 terminal sinks (the
  Rybin instance is one — verified). Kill-path A's search space is small
  out-trees, enumerable exactly, closure by construction.
- gadgets.py (abstract clause systems + exact max-margin LP + NG-3/NG-6
  prunes) and core.check_path_closure (mandatory realization gate) added.
- Enumeration sweep (gadget_enum.py): k=3 floor-size-2 exhaustive → 0
  candidates (extends NG-6 empirically); k=4 coverage minimal — superseded
  anyway by the out-tree reframing, which shrinks the space correctly.

**Ops correction stored (Turso `compute-sizing`, boot-loaded):** pilot before
fleet — the first enumeration launch was a ~4M-call shape with no progress
output that a 900s timeout killed at 100% loss. Estimate = unit-cost ×
count, spoken aloud, before any launch; cheap filters before exact LP;
witnesses before sweeps. muninn-utilities PR #93 added `get(memory_id)`
(the by-ID recall gap that burned calls early in the session).

**Next session (phase 3):** enumerate small out-trees (k = 3..5 terminals,
in-degree-≤2 sinks) with exact coverage/violation search — the first
search space where every verdict is real; extend Lemma 4 toward ≥3 paths
per terminal; decide whether Lemma 1 + Lemma 4 together prove 1.3 for
2-path out-tree instances outright (a theorem-shaped outcome would close
kill-path A for the 2-path grammar and force ≥3 paths).

## Session 2026-07-24 (campaign bootstrap, CCotw, issue branch)

**Gate 1 — literature (CLOSED, green with one strategic downgrade):**
- STVZ Theorem 1.6 verified verbatim: Conj 1.3 ⇒ Conj 1.5 (cost-preserving
  **two-sided** 2·d_max, acyclic). Kill-path B logic VALID, with the note that
  our β* machinery is one-sided (ceilings only): β* > 2 ⇒ no routing has
  (cost ≤ c·x and f ≤ x + 2·d_max) ⇒ a fortiori no routing satisfies 1.5's
  three conditions ⇒ ¬1.5 ⇒ ¬1.3. One-sided is sufficient.
- MS original proves the divisible-demands case (their Thm 4) and a
  (x/2 − d_max, 2x + d_max) bound in general (Thm 5). Acyclicity in 1.3 is
  load-bearing (two-sided fails in non-acyclic graphs even at O(d_max)).
- TVZ Theorem 1.8 (arXiv version): planar + acyclic ⇒ cost-preserving
  two-sided 2·d_max unconditionally. So the β*>2 assassin must be NONPLANAR
  — and doubles as a code check: planar instance with β* > 2 = bug.
  ⚠ Journal version (Math. Prog. 2026) reportedly extends to bounded genus
  with improved planar constants; NOT independently read (paywall served the
  wrong PDF). Treat "nonplanar suffices to dodge TVZ" as needs-reverify
  before any β*>2 claim on a low-genus instance.
- Rybin disproof status: X-thread only (no arXiv as of 2026-07-24), kills
  Goemans/DGG (STVZ 1.2) at 1·d_max. Juang's parametrization of the same
  topology (demands (b+m, b, b+m); Rybin = b=10, m=5) reportedly has
  supremum ~9/8·d_max for the violation threshold on this topology —
  **provisional** (search-snippet sourced, x.com unreachable). If true:
  the known family cannot reach 2, kill-path B needs new structure. → NG-2.
- No post-disproof preprints touching 1.3 as of 2026-07-24. Field clear.

**Task 0 (DONE):** Lemma 1 in NOGOS.md — ceilings-only no-go formalized and
strengthened to any m ≥ 2 chains, k = m+1 terminals, via the
(m−1)·d_1 > m·d_max contradiction. Scope boundary E1–E4 = kill-path A grammar.

**Gate 2 — verifier (this session):** core.py (exact two-sided verifier,
certificate tables) + beta_star.py (breakpoint LP). **Convention correction
against PR #167:** β* = first feasible breakpoint (sup of the infeasible
region), not the last infeasible one; PR #167's engine and its self-hosted
calibration shared the same off-by-one, so its triangle "β* = 1/2" is 1 under
the sup definition, and its family-supremum finding should be re-read
accordingly before reuse. Calibration suite in tests/.

**Rybin instance (RECONSTRUCTED — blocker cleared):** exact topology was
never published (X-thread only); prior sessions were blocked on it. This
session ran an exhaustive constraint reconstruction over the K4-subdivision
class (10,240 orientation candidates → 88 viable topologies → grid over
splits/demand assignments) against all known anchors. Result: unique hit up
to role-labeled isomorphism, all anchors exact (c·x = 58, cost-good =
{ZZZ,EZZ,ZEZ,ZZE} with overshoots 26/16/16/16, β* = 16/15 under the corrected
sup convention, ceiling-good complement costs {60,60,60,90}, cheap mass
16/15, t(x) = −9, t* witness −5 at (1/3, 0, 1/3), Juang b=10/m=5/g=1 at
scale 1/5 gives 58/60). Topology: chain s→a→b→c, terminals as two-parent
sinks t1←{a,c} (15), t2←{c,s} (10), t3←{b,s} (15); costs 2/3/2 on the short
arcs. See rybin.py RECONSTRUCTION_PROVENANCE; 13 exact tests in
test_rybin.py. The 16/15 calibration gate is GREEN for the first time.
Caveat kept honest: t* ≤ −5 (the max side) is grid-certified, not proved.

Structural note for kill-path A: the reconstructed instance's three cheap
paths all share the source arc s→a (source-prefix borrowing in the wild);
x(s→a) = 14 < d_max, so the Goemans-killing mechanism here is pure ceilings
+ costs — consistent with Lemma 1 in that the COST conjecture fell to a
mechanism that cannot, by itself, kill 1.3.

**Kill-path A design notes (grammar, for gadget sessions):** a floor arc
(x(a) > d_max) is a COVERAGE clause: at least one terminal with a path
through a must cross it; if x(a) − d_max > d_i for every single eligible i,
it forces ≥ 2 terminals onto a. Ceilings are CONFLICT clauses ¬(i∧j) at
shared low-x arcs. With 2 paths per terminal, path choice is Boolean, so a
candidate family = a CNF of coverage + conflict clauses realized by a DAG +
consistent x. Serial floor arcs on one corridor are covered together by one
path (x-mass is NOT additive along a path — that's what makes >d_max floors
affordable). Search shape: (1) abstract clause systems that are UNSAT;
(2) LP-check that a consistent x exists at the load level; (3) DAG-realize
or record the obstruction in NOGOS.md.

**Next session:** gadgets.py (kill-path A families per the grammar above);
CEGAR extension of coverage.py for larger routing counts; C7/clique ladder
extension; re-verify the TVZ journal scope + Juang 9/8 cap from primary
sources when reachable.
