# NOGOS.md — killed families and their killing arguments

Campaign ledger for issue #169 (kill Morell–Skutella Conjecture 1.3). Every
family that dies gets an entry with the argument that killed it; the ledger is
the strategic product. Entries are append-only; corrections amend, never delete.

Conventions: demands sorted `d_1 <= d_2 <= ... <= d_k`, `d_max = d_k`, all
demands positive. For a routing `P`, *ceiling at a* is `f^P(a) <= x(a) + d_max`,
*floor at a* is `f^P(a) >= x(a) - d_max`. A floor is non-vacuous only when
`x(a) > d_max` (loads are nonnegative).

---

## NG-1 (Task 0): pure parallel-chain pigeonhole constructions, ceilings only
**Status: KILLED — Lemma 1 below (formalizes the 2026-07-22 session sketch,
and strengthens it: any number of chains, k = m+1 terminals). The parallel
PR #170 session independently proved the m = 2 worst-pair special case via
the conservation floor `x(α)+x(β) ≥ d_2+d_3`; Lemma 1 subsumes it and closes
the "general pairwise-exclusion assignments" question that entry left open
(within hypotheses H1–H3).**

### Setting

Call an instance a *pure parallel-chain construction* with chains
`C_1, ..., C_m` (m >= 2) if:

- **(H1) Full transit / equal tap depths.** Each `C_j` is a directed corridor
  (arc-disjoint from the others), and every s→t_i path that touches any arc
  of `C_j` traverses `C_j` in full. No arc outside the corridors is used by
  paths of two *different* terminals (fold any such shared prefix/suffix into
  a corridor — this is definitional, not a restriction, for architectures
  whose only multi-terminal sharing is the corridors).
- **(H2) Pigeonhole count.** `k = m + 1` terminals.
- **(H3) Full flexibility.** Every terminal has at least one path through
  every chain.

Under (H1), flow conservation at internal corridor vertices makes the
fractional load constant along each corridor: write `X_j = x(a)` for
`a ∈ C_j`. Every unit of every terminal's fractional flow transits exactly
one corridor, so

    X_1 + ... + X_m  =  d_1 + ... + d_k  =: D.                          (1)

A routing induces an assignment σ of terminals to chains; the load on every
arc of `C_j` is `Σ_{σ(i)=j} d_i`, and on an arc private to terminal i it is
`d_i` or `0`.

### Lemma 1 (ceilings cannot do it)

*No pure parallel-chain construction satisfying (H1)–(H3) admits x such that
every routing violates a **ceiling**.*

**Proof.** Private arcs never violate a ceiling: their load is at most
`d_i <= d_max <= x(a) + d_max`. A chain `C_j` occupied by a singleton `{i}`
likewise. So a routing whose assignment puts the two smallest terminals
`{t_1, t_2}` together on chain `C_j` and every other terminal alone on the
remaining `m - 1` chains (exists by (H2)+(H3)) can only violate a ceiling on
`C_j`, which requires

    d_1 + d_2  >  X_j + d_max.                                          (2)

If every routing violates a ceiling, (2) must hold **for every j** (the
isolation assignment exists for each chain). Summing (2) over the m chains
and using (1):

    m (d_1 + d_2)  >  D + m·d_max
                   =  d_1 + d_2 + Σ_{i>=3} d_i + m·d_max
                   >= d_1 + d_2 + (m-1)·d_2 + m·d_max        (d_i >= d_2, i >= 3)

so `(m-1)·d_1 > m·d_max`, impossible since `d_1 <= d_max` and `d_1 > 0`. ∎

The issue's 3-terminal/2-chain sketch is the case m = 2; the sketch's route
through `d_3 > d_2` is replaced by the sharper `(m-1)d_1 > m·d_max`
contradiction, which needs no case analysis and covers all m >= 2.

### Corollary (floors are essential — kill-path A's grammar)

In any counterexample to Conjecture 1.3 of pure parallel-chain shape
(H1)–(H3), pick a pair-isolation routing P from Lemma 1's proof with no
ceiling violation anywhere. Since (G,d,x) is a counterexample, P violates a
**floor**: some arc a has `f^P(a) < x(a) - d_max`, forcing `x(a) > d_max`
(if `f^P(a) = 0` then `x(a) > d_max` directly; if some terminal i uses a then
`x(a) > d_i + d_max`). **Every counterexample in this class contains a
floor-forcing arc `x(a) > d_max`** — an arc that some terminal MUST cover in
every good routing. This is exactly the implication-gadget grammar:
floor ⇒ forced presence ⇒ ceiling elsewhere.

### Exact scope — where the lemma does NOT reach (= design leads)

- **E1 — floors.** The lemma restricts only ceilings. Constructions with
  `x(a) > d_max` arcs are untouched: kill-path A's territory.
- **E2 — unequal tap depths.** If paths enter/leave a corridor mid-way
  (mid-chain taps), x is no longer constant along the corridor, (1) fails
  arc-wise, and the summation breaks. This includes forced-transit
  asymmetries of the Rybin `s→u` kind. OPEN as a ceiling-only design space,
  though any conflict still needs *some* shared arc with
  `x(a) < d_i + d_j - d_max`; per-arc bookkeeping replaces per-chain.
- **E3 — restricted path sets.** Without (H3) the isolation assignment may
  not exist for some chain, and (2) is only forced on chains where it does.
  Terminals with few paths (the Goemans/Rybin instances have exactly 2 each)
  are the natural home of this crack. The per-assignment machinery above
  still generates necessary conditions family-by-family; apply it manually.
- **E4 — k >= m+2 terminals.** Pair isolation needs k-2 <= m-1 singleton
  slots. With more terminals than m+1, every assignment carries >= 2
  co-resident pairs and the weakest-pair argument no longer pins (2) on a
  single chain. OPEN.

Also worth keeping from the PR #170 entry (a demand-vector eligibility fact
usable in any of the cracks): excluding a pair `{i,j}` by ceiling on some arc
requires `x(a) < d_i + d_j − d_max`, so pairs with `d_i + d_j ≤ d_max` cannot
be ceiling-excluded at all (would need negative load). An "every pair
conflicts somewhere" design therefore needs every pairwise demand sum
`> d_max` — e.g. (5,5,6) qualifies, (1,1,10) cannot.

Any proposed ceiling-only family must state which of E2–E4 it exploits;
otherwise it dies by Lemma 1 without further computation.

---

## NG-2: Rybin/Juang K4-subdivision topology as a β* > 2 vehicle
**Status: KILLED (PROVISIONAL — secondary-sourced cap, primary unreachable).**

Kill-path B needs β* > 2. The one known Goemans-killing topology (Rybin's
7-vertex K4 subdivision, demands (b+m, b, b+m); Rybin instance = b=10, m=5,
β* = 16/15) is reported by Juang's parametrization to have supremum ~9/8·d_max
for the forceable violation on this topology — below 2 by a wide margin.
Consequence: kill-path B cannot ride the known family; it needs structurally
different constructions (odd-hole/clique conflict graphs, nonplanar — TVZ
planar 2·d_max makes planar impossible, journal version possibly bounded
genus). *Provisional*: the 9/8 figure came from search-snippet indexing of
x.com/basedjensen (HTTP 402 on direct fetch); re-verify from a primary source
before treating it as a theorem-grade cap. Killing argument if confirmed:
family-level supremum computation on the fixed topology.

---

## NG-3 (design constraint): the parallel floor budget — pure coverage-pigeonhole is impossible
**Status: PROVED (Lemma 2 below). Not a family kill; a grammar constraint
every kill-path A design must respect.**

Call a set of arcs A' an *antichain* (w.r.t. the instance's path system) if no
s→t_i path of any terminal contains two arcs of A'.

**Lemma 2 (antichain mass bound).** For any antichain A',
`Σ_{a∈A'} x(a) ≤ D = Σ_i d_i`.
*Proof.* `x(a) = Σ_i d_i Σ_{p∋a} w_{i,p}`, so
`Σ_{a∈A'} x(a) = Σ_i d_i Σ_p w_{i,p}·|p ∩ A'| ≤ Σ_i d_i Σ_p w_{i,p} = D`,
since `|p ∩ A'| ≤ 1` by the antichain property. ∎

Consequences for floors (`x(a) > d_max` needed for a non-vacuous floor):

- An antichain can carry at most `⌈D/d_max⌉ − 1 ≤ k − 1` floor arcs
  (k terminals, each `d_i ≤ d_max`): each floor needs `x(a) > d_max` and the
  antichain masses sum to at most `D ≤ k·d_max`. A routing supplies k paths,
  which can cover up to k antichain arcs — one each — and only ≤ k−1 floors
  exist. **So coverage can never be blocked by COUNT alone: pure
  floor-pigeonhole ("more parallel floors than the terminals can cover") is
  unrealizable.** Any UNSAT clause system must
  get its contradiction from floor *eligibility* (which terminals have paths
  through the floor arc) interacting with ceiling conflicts — not from
  floor counting.
- *Pair-forcing* floors (`x(a) − d_max` exceeding every single eligible
  demand, so ≥ 2 terminals must cross) cost `x(a) > 2·d_max` when demands
  are ≤ d_max, hence an antichain supports at most `⌈D/(2·d_max)⌉ − 1` of
  them — with 3 terminals, at most ONE. Two parallel pair-forcing floors
  need Σx > 4·d_max ≥ ... > D already at k = 3 (D ≤ 3·d_max). The
  "interleave two implication cycles" design (issue #169) therefore cannot
  use two parallel pair-forcing floors at k = 3; it needs k ≥ 5, or serial
  floors re-used across cycles, or single-coverage floors with tight
  eligibility.

---

## NG-4: abstract odd-hole/clique conflict ladder does not climb toward β* = 2
**Status: STRONG NEGATIVE SIGNAL (grid-limited, not theorem-grade; K5 fine
grid deferred). Session 2026-07-24, ladder.py / LADDER_RESULTS.md.**

Kill-path B's mechanism ladder (conflict graph of cheap options: C3 → C5 →
C7 → K4 → K5, equal and Rybin-shaped demand patterns, exact breakpoint-LP β*
maximized over rational split grids on the abstract pseudo-arc realization):

- Best value anywhere: **β* = 7/6** (C3 equal demands at p = 5/12, and K4
  equal at 5/12 — independently spot-checked exact). Note 7/6 > 16/15: the
  abstract triangle mechanism already beats the realized Rybin value, so DAG
  realization is lossy.
- Larger odd holes go DOWN (C5, C7 ≤ 11/12 on the fine grid); β* does not
  track the stable-set LP gap (K4's gap exceeds C3's; they tie at 7/6);
  every unequal "Rybin-shape" pattern scored ≤ its graph's equal-demand
  baseline.
- Nothing approached 2 on any grid, fine or coarse.

Killing argument (provisional): growing the conflict structure dilutes the
per-arc mass concentration the obstruction needs — the mechanism appears
self-limiting well below 2. If a cap proof materializes (issue #169 names
this outcome "theorem-shaped no-go"), it belongs here; until then the entry
prunes naive "bigger conflict graph" designs. Caveats: abstract pseudo-arcs
(upper bound on the mechanism — realization only lowers β*), grid resolution
1/12, K4/K5 unequal + K5 fine grids deferred (`ladder.py --deep`).

---

## NG-5: K4-parametrized family (1,b,1)/(r,q,r) — sup β* ≈ 1 (CONVENTION-TAINTED, needs re-run)
**Status: CLOSED (session 2) — the corrected convention voids the finding.
The family at b=1, r=q=5/12 is exactly the ladder's C3/equal configuration,
whose β* = 7/6 was verified exactly (independent engine call, this repo).
So the family's sup is ≥ 7/6 > 1: PR #167's "supremum exactly 1, never
attained" was an artifact of the off-by-one convention, not a property of
the family. Superseded by NG-4's mechanism-level 7/6 picture.**

Original entry (kept for the record):

PR #167 found the symbolic family with demands (1,b,1), cheap splits (r,q,r)
has β* supremum "exactly 1, approached but never attained" — under the
sup-convention off-by-one corrected in this repo's `beta_star.py` (β* = first
feasible breakpoint, not last infeasible). Under the corrected convention
every reported value shifts up one breakpoint, so the family's true supremum
is ≥ the reported one and the "never attains 1" reading is void. Re-run the
family through the corrected engine before citing any number from it. Kept as
a ledger entry because the family itself (the natural Rybin generalization)
remains a reasonable baseline; the ladder (NG-4) already bounds its
conflict-graph mechanism class at 7/6 in the abstract.

---

## NG-6 (session 2): single-floor k=3 implication gadgets with full blocking — dead by averaging
**Status: PROVED (Lemma 3 + corollary below). Kills the smallest kill-path A
shape outright and yields a general pruning tool ("blocking cost").**

**Lemma 3 (blocking cost).** Let terminal t (demand `d_t`, exactly 2 paths
p, p') be *fully blocked* against a fixed path q of terminal j (demand `d_j`,
fraction `u` of its split on q): there are arcs `c ∈ p∩q`, `c' ∈ p'∩q` with
ceiling-conflict conditions `x(c) < d_t + d_j − d_max` and
`x(c') < d_t + d_j − d_max`. Since `x(c) ≥ d_t·w + d_j·u` and
`x(c') ≥ d_t·(1−w) + d_j·u` (w = t's split on p; other terminals only add),
summing gives

    d_t + 2·d_j·u  <  2·(d_t + d_j − d_max)
    ⟹  u  <  1 + (d_t − 2·d_max) / (2·d_j).

Full blocking is free only when d_max is small relative to d_t; at
`d_t = d_max` it caps `u < 1 − d_max/(2 d_j)` — the blocked path q cannot
carry a full share of j's split. The split variable of the blocked terminal
t drops out — this is the same summation mechanism as the GPT-trajectory
"3-parallel-gates lemma", now stated as a reusable inequality.

**Corollary (NG-6 proper).** k = 3 terminals, one floor arc F with eligible
set {1,2} entered via paths p₁, p₂ (`x(F) ≥ d_1 w_1 + d_2 w_2 > d_max`
needed), and the design tries to finish by fully blocking terminal 3 against
both p₁ and p₂ (the only UNSAT completion available once F's coverage clause
is (t₁=p₁ ∨ t₂=p₂), since a floor arc can never double as a pair-ceiling:
that would need `d_i + d_j > x(F) + d_max > 2 d_max`). Summing the four
conflict conditions over t₃'s two paths:

    d_1 w_1 + d_2 w_2  <  d_1 + d_2 + d_3 − 2·d_max  ≤  d_max
    (since D = d_1+d_2+d_3 ≤ 3·d_max ⟹ D − 2·d_max ≤ d_max),

contradicting the floor `d_1 w_1 + d_2 w_2 > d_max`. Impossible for EVERY
demand vector. So the minimal implication gadget — one floor, third terminal
conflict-saturated — cannot exist. Any k = 3 single-floor design must leave
some assignment unblocked by conflicts of this form, i.e. k = 3 needs either
two floors (budget: allowed up to 2 by Lemma 2 only if not pair-forcing) or
k ≥ 4.

Design guidance downstream: every candidate clause system should first run
the Lemma-3 summation over each terminal's full path set; if the resulting
demand-only inequality contradicts a floor requirement, the family is dead
without any LP call.

---

## NG-7 (session 2): the declared-menu false positive — and the splice-closure structure theorem
**Status: candidate REFUTED at the certificate gate; Lemma 4 PROVED. The most
consequential entry so far: it redefines kill-path A's search space.**

### What happened

The gadget enumeration surfaced an abstract CANDIDATE (k=3, equal demands,
floors F0={all Z}, F1={Z,Z,E}, conflicts C0={(0,Z),(1,E)}, C1={(1,Z),(2,Z)},
C2={(1,Z),(2,E)}; x-witness w=(1/4,3/8,1/2), margin 1/8) — genuinely UNSAT
with consistent masses (re-verified by hand). A 20-arc DAG realization scored
`t(x)=1/8 > 0` under `verify_counterexample`, "confirmed" by two
independently-written realizations, and was reported as a verified
counterexample to Conjecture 1.3.

**It is not.** `verify_counterexample` enumerates the DECLARED path menus;
Conjecture 1.3 quantifies over ALL s→t_i paths of the DAG. The realization's
merge-diverge sharing creates hybrid paths (prefix of one designated path +
suffix of another through a shared vertex): 4×8×8 = 256 real routings, not
8. Over the full path set, 23 routings satisfy the two-sided bound
(t(x) = −3/8, exact). Both realizations shared the declared-menu blind spot,
so dual-code-path agreement was systemic error — the certificate gate's
adversarial pass caught it. `core.check_path_closure` is now a mandatory
gate before any counterexample verdict on a realized instance.

### Lemma 4 (splice closure ⇒ prefix-only sharing ⇒ out-tree)

Let G be a DAG in which terminal t_i has exactly the declared path set P_i
(closure: declared = all s→t_i graph paths), with |P_i| = 2 for all i. If
paths p (to t_i) and q (to t_j) pass through a common vertex v, the splice
"p's prefix to v + q's suffix from v" is an s→t_j graph path, hence must
equal a declared t_j path. Consequences:

1. **Prefix sharing only.** If p and q share an arc a, they coincide from s
   up to a (else the splice at a's head is a third t_j path).
2. **No reconvergence.** Two paths that diverge never share a later arc
   (a shared later arc means two distinct prefixes into it — splice gives a
   new path).
3. **Out-tree structure.** The union of all declared paths is an out-tree
   rooted at s, except terminals, which are sinks of in-degree ≤ 2 (and a
   declared path passing THROUGH a terminal vertex forces its prefix to be
   a declared path of that terminal).

This is what "splice closure" (the GPT-trajectory realization constraint
quoted in issue #165) actually is. Empirical confirmation: the reconstructed
Rybin instance passes `check_path_closure` and is exactly chain + in-degree-2
terminal sinks — an out-tree in the sense above.

### Consequence for kill-path A

The abstract clause-system framework (gadgets.py) over-generates: sharing
patterns like the refuted candidate's (t1's Z-path meeting t2's Z at C1 and
separately t2's E at C2, after both floors) violate prefix-only sharing and
therefore admit NO path-closed 2-path realization — they were never real
designs. The correct exact search space for 2-path instances is: **out-trees
with ≤ 2k leaves-ish structure (k terminals as in-degree-≤2 sinks), floors
and conflicts as tree arcs, x a per-terminal split over the two root-leaf
routes.** Path closure holds by construction; every abstract verdict is then
a real verdict. Next session: enumerate small out-trees directly. (Instances
with ≥3 paths per terminal escape Lemma 4's scope — richer closure structure,
noted as open.)

---

## NG-8 (session 3): the out-tree reduction, and equal demands are provably safe
**Status: Lemma 5 (reduction) DERIVED + dual-path validated; Lemma 6
(equal-demand no-go) PROVED + stress-tested. Together these cut kill-path A's
search space to unequal-demand out-trees and explain NG-7 structurally.**

### Lemma 5 (reduction to signed tree discrepancy)

By Lemma 4, a path-closed 2-path instance is an arborescence T rooted at s
with each terminal t_i a sink attached from two tree nodes u_i, v_i. Write
z_i = 1 if t_i routes via u_i, w_i for the fractional weight on that path.
For a tree arc a with head-subtree S_a:

    f^P(a) − x(a) = Σ_i C[a,i] · d_i · (z_i − w_i),
    C[a,i] = [u_i ∈ S_a] − [v_i ∈ S_a]  ∈ {−1, 0, +1}.

C[a,i] ≠ 0 exactly when a lies on the tree path between u_i and v_i, signed
by side — so **C is the network matrix of T w.r.t. the terminal chords
(u_i,v_i), hence TOTALLY UNIMODULAR**. Conjecture 1.3 on this class is
exactly: for every w ∈ [0,1]^k there is z ∈ {0,1}^k with

    max_a | Σ_i C[a,i] d_i (z_i − w_i) |  ≤  d_max.

Equivalently: route ξ_i = d_i(z_i − w_i) — chosen from the two-point set
{−d_i w_i, +d_i(1−w_i)}, which straddles 0 with spread d_i — along fixed
tree paths, keeping every arc's net load within d_max. (Pleasingly
self-similar: the conjecture reduces to a rounding problem of its own shape,
but on a tree with no routing freedom, only signs.)

**Terminal arcs caveat (found by cross-checking, worth stating):** terminal
arcs have deviation d_i w_i or d_i(1−w_i), always ≤ d_max, so they can never
violate — but they DO enter the max, so a tree-arcs-only t(w) understates the
true value (Rybin: −11 vs the true −9). The *predicate* t(w) > 0 is decided
by tree arcs alone; the *value* needs both. `outtree.py` exposes both
(`t_tree_of_w`, `t_of_w`) and `t_of_w` agrees with
`core.verify_counterexample` on the realized DAG at every tested w.

**Validation:** the Rybin instance in out-tree normal form (chain s→a→b→c;
attachments (c,a), (c,s), (b,s); demands 15/10/15) reproduces both anchors
through two independent code paths — the C-matrix arithmetic and explicit DAG
path enumeration — t = −9 at its x, −5 at the t* witness.

### Lemma 6 (equal demands ⇒ no counterexample in this class)

*Proof.* [C; I] is TU (C is a network matrix). For w ∈ [0,1]^k the polyhedron
{ζ : ⌊Cw⌋ ≤ Cζ ≤ ⌈Cw⌉, 0 ≤ ζ ≤ 1} has all-integral bounds, hence integral
vertices (Hoffman–Kruskal), and is nonempty (it contains w). Take an integral
z in it: on every row, Cz and Cw lie in the same unit interval, so
|C(z−w)|_∞ < 1 (= 0 where Cw is integral). With all d_i = d = d_max, every
arc deviation is d·|C(z−w)|_a < d_max. ∎

Stress-tested: 24,000 exact evaluations over random equal-demand out-trees,
zero counterexamples, worst t = −1/4 (strictly negative, as the proof
requires).

**Consequences.** (a) The NG-7 false positive used equal demands (1,1,1) —
it could never have been real, independently of the path-closure defect;
two separate reasons, one of them structural. (b) The hunt is confined to
**unequal demands**, and the failure point of the TU argument is exactly the
demand scaling: C is TU, C·diag(d) generally is not. (c) This class-local
result is consistent with Morell–Skutella's own divisible-demands theorem —
a sanity check that the reduction is faithful.

---

## NG-9 (session 3): MSW25 closes every series-parallel out-tree — the live territory is K4-minor chord graphs
**Status: LITERATURE-CLOSED sub-class, with an exact, cheap membership test.
The strongest prune the campaign has produced.**

### The result being applied

Majthoub Almoghrabi, Skutella & Warode, "Integer and Unsplittable Multiflows
in Series-Parallel Digraphs" (arXiv:2412.05182; IPCO 2025 / Math. Prog. 2026)
— **MSW25**, the "series-parallel" reference in issue #169. Theorem 2,
verbatim:

> "The total arc flows (x_e) of a fractional multiflow in a series-parallel
> digraph can be expressed as a convex combination of total arc flows (y_e)
> of unsplittable multiflows that satisfy x_e − d_max < y_e < x_e + d_max for
> every arc e ∈ E."

followed by: *"This result implies, in particular, that Conjecture 1 holds
for series-parallel digraphs, even for general multiflow instances where
commodities have individual source and sink nodes."* Their "Conjecture 1" is
Morell–Skutella's, i.e. our target — and STVZ §1.1 confirms MSW25 proved
even the cost version (Conj 1.4) there. So this is the tight bound, on a
class that permits arbitrary commodity sources/sinks — terminals need not be
special vertices.

### The membership criterion (derived here)

In an out-tree instance every terminal is a sink of in-degree 2, hence an
undirected degree-2 vertex; suppressing it replaces it by the **chord**
{u_i, v_i}. So the instance's underlying graph reduces to **tree + one chord
per terminal**. Therefore:

> **An out-tree instance is MSW25-covered (Conjecture 1.3 provably holds on
> it) iff its tree+chords graph is series-parallel, i.e. K4-minor-free.**

Implemented exactly and cheaply as `outtree.msw25_covered` (classical SP
reduction: delete degree-≤1, suppress degree-2, drop loops/parallels;
K4-minor-free iff it reduces to nothing).

### Why this is the sharpest prune yet

- It **kills the small end of the search space wholesale**. Two crossing
  chords on a path are not enough: path s–a–b–c with chords {s,b}, {a,c}
  reduces to K4-minus-an-edge — series-parallel — so *every* such instance,
  at every demand vector, is proved by MSW25. Counterexamples need chord
  structures rich enough to force a K4 minor.
- **Consistency check that lands hard:** the Rybin instance's chord graph is
  exactly K4 (all six edges on {s,a,b,c}) — *not* covered. The one instance
  known to kill a neighbouring conjecture sits precisely outside the class
  where ours is proved. That is what a meaningful boundary looks like.
- **New self-test**, alongside Lemma 6: a positive t(w) on an MSW25-covered
  instance is a bug, not a discovery. Two independent structural assertions
  now guard the search (equal demands; series-parallel).

### What remains open (the actual hunt)

Instances whose chord graph has a K4 minor AND unequal demands. These are not
covered by MSW25, and not always covered by TVZ's planar 2·d_max result
either — a path a1..a5 with chords on all six non-adjacent pairs suppresses
to K5, which is non-planar, so sufficiently interleaved instances escape both
known theorems. Literature status of this remaining class, checked this
session: **no dedicated tree/arborescence theorem exists** (STVZ §1.1's
survey of known special cases lists only divisible demands, planar 2·d_max,
and series-parallel); and the natural reformulation — linear discrepancy of a
network matrix with **demand-scaled columns** — has no matching theorem and
no matching counterexample. Doerr (Combinatorica 2004) bounds lindisc of
basic TU matrices by 1 − 1/(n+1) but only for **unit** boxes, which is
exactly our Lemma 6 case. The unequal-demand version is genuinely open
territory.

---

## NG-10 (session 3): small out-trees swept — 85% literature-dead, the rest plateaus at −4/9·d_max
**Status: NEGATIVE with explicit coverage bounds. `outtree_search.py`,
`OUTTREE_RESULTS.md`, `outtree_results.json`.**

Enumeration: 16 rooted tree shapes, n = 2..5 nodes (shape counts
cross-checked against OEIS A000081: 1, 2, 4, 9), all attachment patterns for
k = 3,4 (n ≤ 4) and k = 3,4,5 (n = 5), deduped by tree automorphism and
terminal permutation → 27,213 raw / **16,453 deduped structural patterns**.

**MSW25 split (NG-9 applied):** 14,038 covered (**85.3%, provably dead**) vs
2,415 uncovered (**14.7%, live**). Structure of the split, which is the
reusable part:

- **n ≤ 3 can never be uncovered** — the chord graph has only the tree's
  vertices (terminals are suppressed), and a K4 minor needs ≥ 4 branch sets,
  so every instance on ≤ 3 tree nodes is series-parallel regardless of k,
  demands, or attachment multiplicities. Verified exhaustively (705 configs,
  0 uncovered). *The entire n ≤ 3 stratum is dead by vertex count alone.*
- **n = 4**: only 1–6 uncovered patterns per (tree, k) — essentially the
  Rybin K4 completion and its near neighbours.
- **n = 5 chain-shaped trees**: the first genuinely rich live territory (up to
  378 of 2,002 uncovered at k = 5).

**Search result: no positive t(w) anywhere, and no near-miss.** Global best on
the uncovered set: **t\* = −4/3, t\*/d_max = −4/9 ≈ −0.44**, on `n5-chain-fork`
(tree s→n1→n2, n2→n3, n2→n4; attachments (n1,n3), (n3,s), (n4,s); demands
(2,2,3); w = (1/6,1/3,1/3)) — independently re-verified by the orchestrator
through both the C-matrix reduction and `core.verify_counterexample` on the
realized DAG (path-closed, not MSW25-covered). Per-tree bests range −3/2 to
−55/12. For scale: dev/d_max = 0.56 here, against 0.667 for the Rybin
instance and the 0.84 plateau of unstructured random layered DAGs — **small
out-trees are not merely failing to cross the bound, they are further from it
than random instances.**

Self-tests both green: Lemma 6 (equal demands) and the NG-9 MSW25-covered
polarity check, zero assertion failures (197 covered-set checks plus every
uncovered instance).

**Coverage bounds (no silent caps).** n ≥ 6 trees and k > 5 not swept at all.
Within the richest uncovered sets only 15 patterns per combo were sampled
(~4% at the densest combos), 4 demand/permutation trials each. The w-search
is a fine rational grid with polish, not a certified LP optimum — so these are
lower bounds on t\* per instance, not proofs of t\* < 0. Interpretation: the
*stratification* (85% dead by literature, n ≤ 3 dead by structure) is solid
and reusable; the *negative* is strong evidence, not a theorem.

**Where this leaves kill-path A.** The live space is now precisely: chord
graphs with a K4 minor (⇒ n ≥ 4 tree nodes, interleaved chords), unequal
demands, and — since the plateau *falls* as structure shrinks — plausibly
larger n than swept. Two honest next moves: (a) push n = 6,7 on chain trees
with a certified per-instance optimizer (MILP over the w-polytope rather than
grid+polish), or (b) attack the reformulated question directly — linear
discrepancy of a network matrix with demand-scaled columns (NG-9), where the
literature has neither a theorem nor a counterexample. (b) is the better bet:
it is one clean statement, it subsumes every out-tree instance at once, and a
resolution either way is publishable-grade rather than another negative sweep.

---

## NG-11 (session 4): cycle elimination, the interval form, and a certified decision procedure
**Status: Lemma 7 PROVED + verified; Corollary 7.1 and the path/interval
normal form derived; `exists_bad_w` calibrated two-sided as GATE 5.**

### Lemma 7 (cycle elimination)

Let H be the multigraph on tree nodes with one edge {u_i, v_i} per chord. If
the chords with fractional ζ_i contain a cycle in H, then the signed
combination of their columns vanishes identically:
`Σ_j σ_j C[·, i_j] = 0`, because a cycle of chords concatenates their tree
paths into a closed walk, which traverses every tree arc net-zero. So ξ can be
shifted around the cycle **changing no arc load at all**; shift until some ζ_i
reaches 0 or 1 and repeat. Hence:

> **WLOG the fractional chords form a FOREST on the tree's nodes, with every
> arc load unchanged — in particular still all-zero when starting from ζ = w.**

Verified exactly on 233 randomly generated cycles (zero failures), plus a
hand-built example.

**Corollary 7.1.** At most n−1 chords are fractional at a critical point (a
forest on n tree nodes), *independently of k*. With NG-9 (a counterexample
needs a K4 minor, hence n ≥ 4), the smallest possible critical configurations
have 3 fractional chords on 4 tree nodes — which is exactly the Rybin shape.

### Path trees ⇒ interval discrepancy (clean statement of the open question)

Orientation is free (swapping u_i ↔ v_i just relabels z_i ↔ 1−z_i). For a
**path** tree, orienting every chord deeper-end-as-u makes C a 0/1 matrix —
verified — namely the **interval incidence matrix**: column i is the indicator
of the arc-interval between the chord's endpoints. So Q4 restricted to path
trees is exactly:

> Given intervals I_1..I_k on a line and, for each i, a split
> a_i + b_i = d_i with a_i, b_i ≥ 0, can one always choose
> δ_i ∈ {−a_i, +b_i} so that **every** interval sum satisfies
> `|Σ_{a ∈ I_i} δ_i| ≤ d_max = max_i d_i`?

Equal d_i is Doerr's TU/unit-box case (yes, Lemma 6). Non-crossing intervals
are series-parallel (yes, NG-9). The live case is **crossing intervals with
unequal demands** — the same gap the literature gate found unaddressed. Small
sanity: two overlapping intervals are always fine (choose opposite signs, then
|δ_1+δ_2| ≤ max|δ_i| < d_max), so any obstruction needs ≥ 3 mutually crossing
intervals.

### Certified decision procedure (replaces phase 3's heuristic)

`discrepancy.exists_bad_w` decides, per instance, whether ANY w makes every
rounding violate — a MILP over w with per-(routing, arc, side) big-M
disjunctions — and distinguishes **proved** infeasibility from solver error.
Cost ~10 ms at k=5 versus ~0.4 s for phase 3's grid+polish, and it *decides*
rather than lower-bounds.

**Two encoding bugs were caught before use**, both of which would have made
the tool silently answer "no counterexample" everywhere: inverted big-M
polarity (relaxing the *selected* disjunct instead of enforcing it) and a
flipped ε sign. This is why **GATE 5 is two-sided**: it requires the procedure
to *find* bad w below Rybin's known optimum (10) and to *prove* none above it,
with the transition bracketed at 9.75/10.50. A one-sided "no counterexample
found" test would have passed the broken encoding — the same lesson as NG-7,
in a different disguise: a verifier that can only say no is not a verifier.

---

## NG-12 (session 4): the 3/4 plateau — a sharp constant for the whole 2-path grammar
**Status: STRONG CERTIFIED EVIDENCE for a clean conjecture, with an exact
extremal witness. The campaign's best positive-shaped result.**

Using `exists_bad_instance` (which optimizes over **all demands and all w** at
once — see NG-11), define for a tree+chord structure

    R(structure) = sup over demands d and w of
                   [ min over roundings z of max over arcs |dev| ] / d_max.

Conjecture 1.3 fails on the structure iff R > 1.

**Measured: R = 3/4 exactly, everywhere.** Across 702 certified decisions
spanning n = 4..7 tree nodes and k = 3..6 chords, **no structure exceeded
0.75**, and 503/702 (71.7%) attained it exactly. Zero structures exceeded
0.90. Exhaustive over uncovered structures for n ≤ 5, k ≤ 4 (476 structures);
sampled and labelled beyond. An independent adversarial batch by the
orchestrator — deliberately maximally-crossing k=4 chord sets on a 5-node
path, all MSW25-uncovered — also capped at exactly 0.7500.

**The extremal witness (exact, hand-checkable).** The Rybin structure —
chain s→a→b→c with chords (c,a), (c,s), (b,s), so C = [[0,1,1],[1,1,1],[1,1,0]]
— with demands **(1, 1/2, 1)** and **w = (1/2, 1/2, 1/2)**:

| rounding z | max arc deviation |
|---|---|
| (0,0,0), (1,1,1) | 5/4 |
| the other six | **3/4** |

so `min_z max_a |dev| = 3/4` against `d_max = 1`. Verified in exact rationals.
Note this beats Rybin's own demands (15,10,15), which give only 2/3 — the
optimizer found a better demand vector on the same graph, and 3/4 appears to
be the ceiling for the entire class.

### Conjecture 12.1 (ms13-campaign) — **REFUTED at k = 4, see NG-15**

> For every path-closed SSUF instance with exactly 2 paths per terminal
> (equivalently: every out-tree instance, Lemma 4), there is an unsplittable
> routing with `|f^P(a) − x(a)| ≤ (3/4)·d_max` on every arc — and 3/4 is
> attained, by the witness above.

If true, Conjecture 1.3 holds on the entire 2-path grammar **with 25% room to
spare**, and any counterexample to 1.3 must use **≥ 3 paths per terminal** —
precisely the scope boundary Lemma 4 left open. That is a far more useful
statement than "no counterexample found at small size": it is falsifiable by a
single structure with R > 3/4, and provable by an argument that need only
reach 3/4 rather than 1.

**Why 3/4 is plausible as the truth**: equal demands already give < 1 by
Hoffman–Kruskal (Lemma 6); the extremal witness needs *unequal* demands in the
exact ratio 2:1:2 and all-fractional w = 1/2, i.e. maximal frustration in both
axes simultaneously; and the value did not move at all as n grew from 4 to 7,
while the *live* (MSW25-uncovered) fraction of structures grew from 15% to
31%. More freedom did not buy a higher ratio.

### Honest bounds

n ≥ 8 and k ≥ 7 untouched; n = 6,7 at k = 5 mostly failed to resolve — a
genuine branch-and-bound pathology in `exists_bad_instance` at k ≥ 5
(>160s), not a sampling choice. `sup_ratio.py` wraps calls in a hard-killed
subprocess and records kills as `solver_error` rather than dropping them. So
Conjecture 12.1 is *strongly evidenced, not proved*; the natural attack is the
interval form (NG-11) where the extremal witness is only 3 chords wide.

---

## NG-13 (session 5): k=3 is a finite case analysis — lower bound PROVED, R ≥ 3/4 attained
**Status: LOWER BOUND PROVED (hand proof + exact verification). Upper bound
under exact case analysis (`k3_proof.py`, `K3_PROOF.md`). This is the first
part of Conjecture 12.1 to become a theorem.**

### Why k=3 is finite

R depends on a structure only through its **set of rows** `C[a,·] ∈ {−1,0,1}^k`,
modulo row sign flips (the objective is `|·|`), column sign flips (chord
reorientation, which also swaps p_i ↔ q_i — a symmetry of the normalized box),
and column permutation (chord relabelling, which permutes demands). R is
**monotone non-decreasing** under adding rows, so the maximum over all
structures is attained at *maximal* row-sets.

Enumerating all trees on n ≤ 6 nodes exhaustively, plus 180k random configs at
n = 7,8,9, yields **25 distinct k=3 types** (two first appearing at n = 7, none
new at n = 8,9 under sampling). Measured R takes **only three values: 1/2, 2/3,
3/4** — quantized, not a continuum.

### Lemma 8 (lower bound, PROVED)

The minimal type attaining 3/4 has just **three rows**, and it is the Rybin
type: normalized, `{(1,1,1), (1,1,0), (1,0,1)}`. Write column 1 for the chord
present in every row. The three constraints are

    |δ_1 + δ_2 + δ_3| ,  |δ_1 + δ_2| ,  |δ_1 + δ_3|   ≤  B.

Take `p_i = q_i = d_i/2` (i.e. w ≡ 1/2), `d_1 = t`, `d_2 = d_3 = 1`, so
`δ_1 = ±t/2` and `δ_2, δ_3 = ±1/2`. Two cases:

- **δ_2, δ_3 opposite signs** ⇒ `δ_2+δ_3 = 0`, so the triple row gives `|δ_1| =
  t/2`, but the two pair rows give `|δ_1 ± 1/2|`, whose max is `t/2 + 1/2`.
- **δ_2, δ_3 equal signs** ⇒ `|δ_2+δ_3| = 1`; choosing δ_1 opposite gives max
  `1 − t/2`.

So `min over roundings = min(t/2 + 1/2, 1 − t/2)`, which is maximized where the
branches cross, at **t = 1/2, value 3/4**. Hence `R ≥ 3/4`, attained. ∎

Verified exactly: the formula reproduces the computed min-max at
t = 1/4, 1/2, 3/4, 1 (5/8, 3/4, 5/8, 1/2), and the t = 1/2 instance is exactly
the witness the demand-quantified MILP had found independently in NG-12
(demands (1, 1/2, 1) with w ≡ 1/2 on the Rybin structure). Two independent
derivations, one symbolic and one by optimization, agreeing on the extremal
point.

**Note the shape of the extremal example**: it needs the "hub" chord (the one
on every arc of the overlay) to carry *half* the demand of the other two, and
all three splits exactly fractional. That is maximal frustration in both axes
at once, and it explains why equal demands (Lemma 6) and non-crossing chords
(NG-9) both miss it.

### Quantization made exact (lower-bound side)

The per-type R values in NG-12 were float MILP outputs. Every one has now been
rationalized and re-verified in exact arithmetic: all 24 decidable k=3 types
have an **exact rational witness (d, w)** attaining precisely the float value —
**12 types at 3/4, 8 at 2/3, 4 at 1/2**, and nothing else. So the quantization
`R ∈ {1/2, 2/3, 3/4}` is exact as a set of *attained* values; only the ceiling
(no type exceeds 3/4) still rests on the upper-bound analysis.

### NG-13a: extending the extremal pattern to k=4 does not help
**Status: NEGATIVE (targeted, 260 structures). CAVEAT (NG-15): this tested only
structures CONTAINING the k=3 extremal pattern; k=4 structures outside that
slice reach 4/5, so do not read this entry as evidence about all of k=4.**

The obvious way to beat 3/4 is to take the extremal k=3 configuration and add
a chord. Enumerating k=4 structures on n = 4..6 whose row-set *contains* the
extremal 3-row pattern on some triple of columns, restricted to
MSW25-uncovered ones: **260 decided, max R = 0.750000, none exceeding it.**
The fourth chord buys nothing — consistent with the phase-4 observation that R
*decreases* with k (the k=6 spot check gave 2/3). Evidence that 3/4 is not just
the k=3 maximum but the global one, and that the extremal configuration is
essentially 3-terminal.

### Upper bound: R ≤ 3/4 — PROVED exactly (Lemma 9)

`R ≤ v` for a row-set is equivalent to: the 8 polytopes
`P_z = {(p,q) in the normalized box : all |loads| ≤ v}` **cover** the box. The
negation is a disjunctive system (per rounding, pick a strictly violated
(row, side)), searched depth-first with an **exact Fraction LP** at each node
and the margin maximized symbolically — no numeric epsilon anywhere.

**There are exactly two maximal types**, computed symmetry-aware over the
48-element group (`3! × 2³`); both have 6 rows. Both are PROVED:

| maximal row-set | verdict | nodes | exact LP calls | time |
|---|---|---|---|---|
| `(-1,0,0),(-1,1,-1),(-1,1,0),(0,-1,0),(0,-1,1),(0,0,-1)` | **R ≤ 3/4** | 15,975 | 191,700 | 47.6 s |
| `(-1,0,0),(-1,0,1),(-1,1,0),(0,-1,0),(0,-1,1),(0,0,-1)` | **R ≤ 3/4** | 14,440 | 173,280 | 40.7 s |

Zero leaves survived either search. By monotonicity this bounds every type.
Combined with Lemma 8: **R = 3/4 exactly, for every k = 3 out-tree structure.**

**Prover validated two-sided.** On the minimal 3-row extremal type it proves
`R ≤ 3/4` (1,866 nodes) and **correctly fails** at `R ≤ 5/8` — which must fail,
since 3/4 is attained. A prover that only ever says "proved" passes the first
test and is worthless; this is GATE 5's discipline applied to the proof itself.

### Completeness of the census (the one real caveat, and how it was closed)

Two independent enumerations, reconciled: an exhaustive isomorphism-deduped
*tree* enumeration through n = 7 gives **25** classes, while a
**reduced-object** enumeration (tree shape + placement of the 6 chord
endpoints, coincidences allowed) gives **30** — a strict superset, the 5 extras
being degenerate coincidence cases the first method's validity filter excluded.
Neither found a class the other missed beyond that. **All 30 classes from both
censuses are dominated by the two proved maximal types.**

**CLOSED (session 5, later the same day).** The census is now exhaustive
through **m = 10** — the completeness bound — and found **no new type beyond
m = 7** (m = 8, 9, 10 each: 0 new; 30 total). The two maximal types it yields
are, verified programmatically, **exactly the two already proved** in Lemma 9.
So `R = 3/4` at k = 3 is **unconditional**. The closure needed a faster shape
generator: the original brute-forced all `m^(m-2)` Prüfer sequences (100M at
m = 10, ~50 min projected); leaf-extension generation (every free tree on m
nodes is one on m−1 plus a pendant leaf, deduped by AHU signature, validated
against A000055 = 1,1,1,2,3,6,11,23,47,106) does the whole census in 104 s.
See `fast_census.py`, `k3_census_complete.json`, `K3_PROOF.md` appendix.

The reduced-object method also supplies the completeness *argument*:
subdividing an edge at an unmarked degree-2 node duplicates a row rather than
creating a new one, and since R quotients by per-row sign flip the root's
placement is irrelevant — so with only 6 marked nodes every unmarked node needs
degree ≥ 3, giving ≤ 4 internal nodes and **n ≤ 10 suffices**. Exhaustive
through n = 9 found no new type after n = 7 (n = 8: 0 new; n = 9: 0 new); the
n = 10 closure run is recorded in `K3_PROOF.md`.

### NG-13b: no simple rounding rule certifies 3/4
**Status: NEGATIVE, and it redirects the general-k strategy.**

Four candidate certifying rules, all exact: decreasing-demand greedy min-max
(**fails on 17/30** types), decreasing-`max(p,q)` (11/30), oppose-current-load
(**30/30**, worst ratio 3/2 — actively harmful), small-side-first (17/30).
Tightest failure: on `{(0,0,1),(0,1,-1),(1,-1,1),(1,0,1)}` at
`p = q = (1/2,1/2,1/2)` the greedy rules commit to `z = (1,1,1)` for
`max|load| = 1`, where the true optimum is `1/2` at `z = (0,1,1)` — committing
the largest-demand chord first is provably the wrong move.

Interpretation: the extremal configurations are **genuinely simultaneous**
across all three chords, so no scalar processing order separates them into a
safe sequence. Consequence: the exact case analysis is currently the only
working route, and a constructive general-k proof needs something beyond simple
greedy (two-pass or backtracking rules remain untested). This is the useful
kind of negative — it eliminates the obvious proof strategy rather than just
failing to find a counterexample.

---

---

## NG-14 (session 6): k=4 — the forest reduction, and why the theorem does NOT close here
**Status: Lemma 10 PROVED (a real reduction). Full k=4 theorem is a documented
COMPUTE NO-GO, with the arithmetic below. Partial census in progress.**

### Lemma 10 (forest reduction for R)

Apply cycle elimination (Lemma 7) to R itself. If the chord multiset C contains
a cycle in H, shift ξ around that cycle: **no arc load changes**, so starting
from ζ = w (all loads 0) we reach a point where some chord is integral and the
loads are still 0. That chord's contribution is now fixed and already baked into
the zero load vector, so the remaining task is *the same problem* on the
strictly smaller chord set — same demands, and d_max unchanged (hence the
target (3/4)·d_max is, if anything, more generous for the sub-instance).
Iterating until no cycle remains:

    R(C)  ≤  max over forest sub-multisets F ⊆ C  of  R(F).

**Consequence for k = 4:** any 4-chord set containing a cycle in H reduces to a
forest with ≤ 3 chords, which the k = 3 theorem (NG-13, unconditional) already
bounds by 3/4. **So only 4-chord sets that are FORESTS in H require new
analysis.** Same argument caps every k: non-forest configurations never need
their own treatment.

### Why the k=4 theorem does not close (sizing, done before spending compute)

The completeness bound scales with the marker count: 4 chords ⇒ 8 marked nodes
⇒ at most 6 unmarked internal nodes of degree ≥ 3 ⇒ **n ≤ 14** (versus n ≤ 10 at
k = 3). Census size, configs = free_trees(m) × multisets of 4 node-pairs:

| m | shapes | marker multisets | configs |
|---|---|---|---|
| 8 | 23 | 31,465 | 723,695 |
| 10 | 106 | 194,580 | 20,625,480 |
| **14 (the bound)** | 3,159 | 3,049,501 | **9,633,373,659** |

Cumulative to m ≤ 14: **1.24 × 10^10 configs**. The canonicalization group is
now `4! × 2^4 = 384` elements (8× the k = 3 group), measured at ~0.6 ms/config,
giving **≈ 2,070 hours** for the census alone — four orders of magnitude past any
session budget.

The exact branch-and-bound is worse, not better. At k = 3 it searched 8 rounding
levels and used 15,975 nodes / 191,700 exact LP calls / 48 s on a 6-row type. At
k = 4 there are **16** levels and more rows per type, so both depth and breadth
grow; even assuming the same pruning efficiency the node count is plausibly
~10^8, i.e. days per type at the measured ~330 nodes/s. `exact_upper_bound_bb`
is also hardcoded to k = 3 (`_ZS`, `_VARN`) and would need generalizing first.

**Verdict:** k = 4 is not closable as a theorem with this machinery. What is
achievable is a *forest-only* census at small m plus certified-per-instance
float MILP screening — strong evidence, of the same kind NG-12 gave, not proof.
Recorded here so a future session does not rediscover the wall by grinding into
it; the cheap arithmetic above is the whole reason to write it down.

---

---

## NG-15 (session 6): **Conjecture 12.1 is REFUTED at k = 4.** R = 4/5, and the pattern is k/(k+1)
**Status: REFUTATION VERIFIED EXACTLY. This overturns NG-12's central guess and
replaces it with a better one.**

### The refutation

A k = 4 forest row-set from the m = 7 census,

    {(0,0,0,1), (0,0,1,-1), (0,1,-1,0), (1,-1,1,-1), (1,0,0,-1), (1,0,1,-1)}

with **all demands equal to 1** and

    p = (3/5, 3/5, 2/5, 1/5),   q = (2/5, 2/5, 3/5, 4/5)   i.e. w = (3/5, 3/5, 2/5, 1/5)

has `min over the 16 roundings of max over rows |load| = 4/5` **exactly**
(verified at denominators 5, 10, 20, 60 — a stable exact rational, not a float
artifact). Since 4/5 > 3/4, **Conjecture 12.1 is false.**

Conjecture 1.3 is *not* refuted: 4/5 < 1, so this instance still admits a
routing inside ±d_max. The 2-path grammar remains safe — with a worse constant
than claimed.

### What NG-12 got wrong, and why

NG-12 measured R = 3/4 across 702 certified decisions and read the k-trend as
*decreasing* (a k = 6 spot check gave 2/3). That reading was an artifact of
coverage, not a fact:

- the phase-4 sweep's k = 4 stratum was **sampled**, and its uncovered set was
  explored only ~4% at the densest combos;
- **NG-13a**, the targeted k = 4 attack, only tested structures **containing the
  extremal k = 3 pattern** — a narrow slice. Those all capped at 0.75, and I
  over-read that as evidence about all of k = 4. The refuting row-set does *not*
  contain the k = 3 extremal pattern.

Both prior entries stand as records of what was measured; the *inference* from
them to "3/4 is the global max" is retracted here.

### The k=4 extremal needs no unequal demands

Note the witness has **all demands equal**. At k = 3 the extremal required the
unequal ratio 2:1:2 (Lemma 8), and Lemma 6 (Hoffman-Kruskal) guarantees only
`< 1` for equal demands — never `≤ 3/4`. So equal demands were never protected
at 3/4, and at k = 4 they already beat it. Lemma 6 is the only thing bounding
this family, and it bounds it by 1.

### Conjecture 12.2 (replacement)

> `R_max(k) = k/(k+1)`, attained. Hence `sup over all k = 1`, approached but
> never attained.

Data: k = 3 → 3/4 (**proved**, NG-13, both bounds exact); k = 4 → 4/5 (lower
bound **proved exactly** here; the matching upper bound is not proved).
Prediction for k = 5: **5/6 ≈ 0.8333** — a sharp, cheap test for the next
session.

If 12.2 holds, the consequence is much more interesting than 12.1 was:
**Conjecture 1.3 is TIGHT on the 2-path grammar in the limit** — instances exist
whose best routing deviates by `(1 - 1/(k+1))·d_max`, approaching the bound
arbitrarily closely, while never crossing it. That would explain why the class
resisted every counterexample search *and* why no uniform margin was ever found:
there isn't one.

### Addendum: k=5 probe INCONCLUSIVE, and R_max is monotone in k

A bounded random probe of k = 5 forest structures decided only **5 structures in
655 s** — the row-set MILP costs ~130 s per structure at k = 5 (32 rounding
levels vs 16 at k = 4). Max observed 0.800000. **Five samples is not a test**:
this neither confirms nor refutes 12.2's prediction of 5/6, and it should not be
cited either way. Deciding k = 5 needs the MILP replaced (CEGAR over roundings,
or a time-limited warm-started solve — cf. the k >= 5 pathology already recorded
in NG-12).

**Observation (trivial but clarifying): `R_max(k)` is non-decreasing in k.**
Demands may be zero (`p_i = q_i = 0` is in the normalized box), and a chord with
`d_i = 0` contributes nothing to any load, so any (k-1)-chord instance is a
degenerate k-chord instance. Hence `R_max(k) >= R_max(k-1)`, `sup_k R_max(k)` is
well-defined, and it is `>= 4/5` by NG-15. Finding 0.8 among 5 random k = 5
structures is therefore unsurprising — it is the k = 4 value showing through,
not evidence about k = 5's maximum.

This also sharpens what is actually at stake: **`sup_k R_max(k) = 1` iff
Conjecture 1.3 is tight on the 2-path grammar** (no uniform margin), and
`sup_k R_max(k) < 1` would give a uniform margin and a strictly stronger theorem
than 1.3 on this class. Conjecture 12.2 asserts the former, with the specific
rate k/(k+1).

### Method note (a real search-quality lesson)

My own exact integer hill-climb on this row-set found only **7/10**, below 3/4 —
a local optimum on the piecewise-linear landscape, with 400 restarts and single
-coordinate ±1 moves. Had I trusted it, I would have "confirmed" the wrong
answer and left 12.1 standing. The MILP's witness, rationalized and checked by
an independent exact evaluator, is what settled it. Same lesson the discrepancy
experiment recorded (`restart diversity >> iteration depth`), and the reason the
rule here is *verify the witness*, never *trust the search*.

---

---

## NG-16 (session 6, strategic): k/(k+1) IS Doerr's TU bound — the campaign was rediscovering a known theorem
**Status: REFRAMING. Turns an exponentially-growing search into one inequality.
Prompted by Oskar's observation that enumeration cannot scale and "an entirely
different approach is needed" — correct, and the different approach was already
in our own phase-3 literature notes, mis-filed.**

### The identification

`k/(k+1) = 1 − 1/(k+1)` is exactly the form of **Doerr's bound for the linear
discrepancy of totally unimodular matrices with k columns** (Combinatorica 2004,
`lindisc(A) ≤ 1 − 1/(n+1)`), which the phase-3 gate found and NG-9 filed away as
"covers only the unit-box (equal-demand) case." That was the answer, not a
footnote. Our measured maxima, all exact:

| k | unit-box lindisc (all d_i = 1) | max with demand scaling | 1 − 1/(k+1) |
|---|---|---|---|
| 2 | **2/3** | 2/3 | 2/3 |
| 3 | 2/3 | **3/4** (needs hub demand 1/2) | 3/4 |
| 4 | **4/5** (equal demands suffice) | 4/5 | 4/5 |

Note the two regimes genuinely differ: at k = 3 attaining the bound *requires*
unequal demands; at k = 4 equal demands suffice (and that instance is literally a
unit-box lindisc computation). So column scaling is part of the phenomenon, not a
side condition — but it never *exceeds* the unit-box bound's value.

### The open problem, restated as ONE inequality

> **Question Q7.** For a network matrix C with k columns and demands d with
> `d_max = max_i d_i`, is `lindisc(C·diag(d)) ≤ d_max·(1 − 1/(k+1))`?

**If yes, Conjecture 1.3 holds on the entire 2-path grammar**, because
`1 − 1/(k+1) < 1` for every finite k — no enumeration, no per-k grind. The
extremal witnesses then make it tight, so `sup_k = 1` and 1.3 is *saturated but
never violated* on this class. If no, the counterexample is a counterexample to
1.3 itself.

### Why enumeration was always going to fail (recorded so it is not retried)

The completeness bound grows with k (roughly 4k − 2 nodes: 2k markers plus
≤ 2k − 2 unmarked branch nodes), so census size grows like
`trees(m) × C(m²/2, k)` **and** the per-instance decision cost grows as `2^k`
rounding levels. Measured: k = 3 closed in 104 s; k = 4 projected ~2,070 h for the
census alone (NG-14); k = 5 could not decide even 5 structures in 655 s (NG-15
addendum). Enumeration yields data points, never the theorem. It did its job —
it produced the three exact values that identified the pattern — and that job is
finished.

### Phase-7 plan (three tracks, none exponential)

1. **Targeted literature check** on the column-scaled generalization
   specifically ("weighted/box linear discrepancy of TU matrices",
   `lindisc(AD)`, Doerr's own follow-ups, LSV-style transfer arguments). Q7 may
   simply be known. Also: get Doerr's exact hypotheses ("basic" TU — does it
   cover network matrices?) and his tight family, which would settle our
   lower bound for all k at a stroke.
2. **Construct the extremal family** for all k (we have exact witnesses at
   k = 2, 3, 4). This is algebra, not search: a parametrized family attaining
   `1 − 1/(k+1)` proves the lower-bound half for every k and hence `sup = 1`.
3. **Prove Q7** (or refute it). A single inequality about TU matrices, provable
   by induction / polyhedral argument, entirely independent of tree enumeration.
   Lemma 7's forest reduction and Lemma 10 remain available as structural tools.

---

---

## NG-17 (session 6): the extremal family, PROVED for all k — and Doerr 2004 already owns the equal-demand case
**Status: LOWER BOUND OF 12.2 PROVED FOR EVERY k (3-line proof + verified
k=2..7). Equal-demand upper bound is a known theorem. Q7 (column-scaled)
confirmed genuinely open.**

### Theorem 11 (the extremal family) — **PRIOR ART, see NG-20. Independently proved here, but published in Morell–Skutella 2022 Fig. 3.**

**Construction.** Tree `s → c`, then `c → b_i` for i = 1..k (a spider). Terminal
`t_i` attaches at `b_i` and at `s`. In SSUF terms: **terminal t_i has a long path
`s→c→b_i→t_i` and a short path `s→t_i`** — the simplest two-path gadget there is.
Row-set: one all-ones row (arc `s→c`, traversed by every chord) plus k singleton
rows (arc `c→b_i`, private to chord i).

**Take all demands 1 and `w_i = k/(k+1)` on the long path.** Then
`δ_i ∈ {+1/(k+1) (long), −k/(k+1) (short)}`, and:

- if every `z_i` = long: the all-ones row carries `Σδ_i = k/(k+1)`, each singleton
  row carries `1/(k+1)`. Max = **k/(k+1)**.
- if any `z_j` = short: that terminal's *private* row carries
  `|δ_j| = k/(k+1)`. Max ≥ **k/(k+1)**.

Every rounding therefore has max deviation ≥ k/(k+1), and the all-long rounding
attains it. So `R = k/(k+1)` **exactly, for every k**. ∎

Verified exactly for k = 2..7 (2/3, 3/4, 4/5, 5/6, 6/7, 7/8), each instance
valid, path-closed, and matching the predicted value. **This proves the
lower-bound half of Conjecture 12.2 for all k**, hence `sup_k R_max(k) = 1`.

### Consequence: Conjecture 1.3's constant is optimal on this class

For any ε > 0, choose k with `k/(k+1) > 1 − ε`: every routing of that instance
deviates by more than `(1−ε)·d_max` somewhere. So **no bound of the form
`(1−ε)·d_max` holds uniformly** on the 2-path grammar — 1.3's `d_max` is the
right constant, approached but never reached. This is exactly the "saturated but
never violated" picture NG-15 conjectured, now proved on the lower side.

Sharper still: **the family is series-parallel** (`msw25_covered` = True for
every k), so MSW25's theorem already proves 1.3 on it — meaning **MSW25's
`d_max` bound is itself asymptotically tight, on the simplest possible gadget.**

⚠ *Novelty caveat*: this construction is elementary and may well be folklore —
tightness of the `d_max` constant is the sort of remark a paper makes in
passing. It was NOT specifically searched for. Before treating Theorem 11 as
new, search "tightness of d_max unsplittable flow", Morell–Skutella §1, and
DGG's original discussion. The *proof* is ours and verified; the *priority* is
unchecked.

### Doerr 2004 owns the equal-demand upper bound

The phase-3 gate had filed Doerr as "basic TU only" — the EJC 2000 paper does
require ≤ 2 nonzeros per row, which network matrices violate. But the
**Combinatorica 2004** paper ("Linear Discrepancy of Totally Unimodular
Matrices", 24:117–125) drops "basic" and gives, for **general** TU A:

> `lindisc(A) ≤ 1 − 1/(n+1)`, and this bound is sharp; also `≤ 1 − 1/m` for
> m ≥ 2. Tight matrices are characterised: m×1 single-nonzero, or those
> containing n+1 rows every n of which are linearly independent.

Network matrices are TU, so **the equal-demand (unit-box) case of Q7 is a known
theorem**, and NG-15's k = 4 discovery (equal demands, 4/5 = 1 − 1/5) was an
instance of its tightness — we rediscovered a 2004 result empirically. Our
spider family satisfies the tightness characterisation by inspection: its k+1
rows (all-ones plus k singletons) have every k of them linearly independent.

⚠ Springer paywall blocked the full text; the statement is corroborated by
consistent independent snippets but is **not primary-verified**. Get the actual
paper before leaning on the exact constants.

### Q7 remains open, with a proof lead

The **column-scaled** inequality `lindisc(C·diag(d)) ≤ d_max·(1 − 1/(k+1))` was
not found anywhere, under any framing tried (weighted linear discrepancy, box
discrepancy, `lindisc(AD)`, vector balancing with unequal norms). Nearest
relative: TVZ's "Weighted Partition-Constrained Selection" (arXiv:2308.02651
Thm 2.10) proves a **flat** `d_max` bound for non-interleaving circular-interval
structures — same `d_max` currency, weaker constant, and an equality-constrained
regime rather than an independent box per column.

**Proof lead (from the literature scan):** Doerr's own EJC 2000 combinatorial
method — partition columns by weight into "extreme" (near 0/1) versus "moderate"
classes via Ghouila-Houri, round the extremes to the nearest integer and balance
the moderates by hereditary discrepancy ≤ 1 — is more naturally extensible to
per-column widths `d_i` than the LSV/Matoušek geometric argument
(`lindisc ≤ 2·herdisc` via `U_A = {x : ‖Ax‖_∞ ≤ 1}`), because it already reasons
about individual column weights against a partition of [0,1]. Making that
partition `d_i`-dependent is the natural attack. Nobody appears to have done it.

---

---

## NG-18 (session 6): Q7 restated — "does column scaling exceed Doerr's bound?" — and the k=3 theorem answers it for k=3
**Status: REFRAMING + a claim of contribution to discrepancy theory, stated with
its priority caveat. This is the sharpest form of the remaining question.**

### The restatement

Two facts now sit together:

1. The spider family (Theorem 11, NG-17) attains `1 − 1/(k+1)` at **equal**
   demands, for every k.
2. Doerr 2004 bounds `lindisc(A) ≤ 1 − 1/(n+1)` for **general TU** A, sharply —
   and network matrices are TU, so the equal-demand case is settled, with the
   spider as a tight example.

So the entire open question collapses to one thing:

> **Q7′. Can column scaling exceed the unit-box bound?** I.e. for a network
> matrix C with k columns and demands d, is
> `lindisc(C·diag(d)) ≤ d_max·(1 − 1/(k+1))`, the same constant as the
> unscaled case?

This is sharper than Q7 as first posed (NG-16), because it identifies exactly
what is unknown: not the value of the bound — Doerr gives that — but whether
*unequal* column widths can beat it. Note the answer is not obvious in either
direction, and the naive intuitions both fail:

- *"Unequal demands weaken the adversary"* — false. At k = 3 the Rybin-type
  structure gives only 2/3 at equal demands but reaches **3/4** with the hub
  demand halved (Lemma 8). Scaling genuinely helps there.
- *"So scaling should push past the bound"* — also false, at least at k = 3
  (below).

### Our k=3 theorem IS Q7′ for k=3

NG-13's theorem — `R = 3/4` exactly for **all** 3-chord out-tree structures and
**all** demand vectors, both bounds proved, census complete through m = 10 —
is precisely the statement `lindisc(C·diag(d)) ≤ d_max·(1 − 1/4)` for network
matrices with k = 3, attained. That is **the first non-trivial case of the
column-scaled generalization of Doerr's theorem**, and it is the one genuinely
new-looking mathematical object the campaign has produced.

Modest, and stated with the caveat: k = 3 is small, the proof is a finite case
analysis (30 row-set types, exact branch-and-bound on the 2 maximal ones), and
it gives no technique that extends. But it is a real answer to an open question
that the literature does not address, rather than a rediscovery.

### Status of the two halves, honestly

| statement | status |
|---|---|
| equal demands, any k | **known** (Doerr 2004, general TU, sharp) |
| any demands, k = 3 | **proved here** (NG-13) — new as far as the literature scan reaches |
| any demands, k = 4 | consistent (max 4/5 observed, NG-15) — unproved |
| any demands, general k | **OPEN** = Q7′ |
| lower bound `1 − 1/(k+1)` attained for all k | **proved here** (Theorem 11, NG-17) — but possibly folklore, priority unchecked |

Note the campaign's headline result for *flow* (1.3 is tight on the 2-path
grammar) follows from Theorem 11, whose construction is elementary enough that
prior art is likely; whereas the result least likely to be known (k = 3 column
scaling) is the one that sounds least impressive. That asymmetry is worth
remembering when deciding what to write up.

### Attack on Q7′ for general k — status: NOT ATTEMPTED BEYOND THE LEAD

The technique lead from the literature scan stands: Doerr's EJC 2000 method
partitions columns by weight (extremes near 0/1 versus moderates) using
Ghouila-Houri, then rounds extremes and balances moderates via hereditary
discrepancy ≤ 1. It reasons about individual column weights against a partition
of [0,1], so making that partition `d_i`-dependent is the natural generalization.
Also available: Lemma 7's forest reduction and Lemma 10, which mean any proof
may assume the chords form a forest. No proof attempt has been made; this is
where a next session should start, and it is mathematics, not compute.

---

---

## NG-19 (session 6): **Conjecture 13 — the k+1 staircase roundings suffice. REFUTED at k=8.**
**Status: FALSE. Adversarial MILP search found staircase-best deviations up to
1.636·d_max at k = 8 (equal demands). The random-instance evidence below
(5,389 instances, k ≤ 6, plus every known extremal) was a small-k artifact — the
THIRD time in this campaign that a k ≤ 6 trend failed to survive larger k.
Refuted BEFORE any proof effort was spent, by deliberately searching for the
counterexample first. See the refutation section at the end of this entry.**

### Statement

Given a network matrix C (k columns), demands d, and w ∈ [0,1]^k: sort the
coordinates by **w descending** and define the k+1 **staircase roundings**
`z^(0), …, z^(k)`, where `z^(j)` rounds up exactly the j largest w-coordinates.

> **Conjecture 13.** `min over j ∈ {0..k} of max over arcs a of
> |Σ_i C[a,i]·d_i·(z^(j)_i − w_i)| ≤ d_max.`

I.e. one need never search 2^k roundings — **k+1 sorted candidates always
suffice** to meet Conjecture 1.3's bound.

### Why this is the right shape of answer

Consecutive staircases differ in exactly one coordinate, so each arc's load moves
by `C[a,i]·d_i ∈ {−d_i, 0, +d_i}` — **steps of magnitude ≤ d_max**, which is
precisely the granularity the bound asks for. That is not a coincidence; it is
why the family is the natural candidate.

**Partial proof (monotone rows).** If a row's nonzeros all share one sign, the
staircase sequence is *monotone*, stepping by exactly `d_i` from
`−Σ_i C[a,i] d_i w_i` (all-down) to `+Σ_i C[a,i] d_i (1−w_i)` (all-up). It
therefore crosses zero, and since consecutive values differ by at most `d_max`,
some consecutive pair straddles 0 — giving `|L_a| ≤ d_max/2` at one of them.
**What is open** is *simultaneity*: a single j that works for every row at once,
and mixed-sign rows, whose sequences are not monotone.

### Evidence

- **5,389 random instances** (trees n = 3..8, k = 2..6, rational demands and w
  over several denominators, exact arithmetic): the best staircase rounding
  **never exceeded d_max**. Worst ratio observed exactly 1.
- **Every known extremal instance is attained exactly** by a staircase:
  the spider family at k = 2..7 (2/3, 3/4, 4/5, 5/6, 6/7, 7/8 — all matched),
  the k = 3 Lemma-8 extremal with unequal demands (3/4), and the k = 4 refuting
  type from NG-15 (4/5). The family is optimal precisely on the instances that
  matter.
- It is *not* optimal in general — it matched the true optimum in only 82.9% of
  random instances (worst gap: staircase 1 vs optimum 1/6). So Conjecture 13 is
  strictly weaker than "staircase = optimum", and that is fine: it needs only to
  clear d_max, not to be optimal.

### Why it matters more than anything else in the ledger

Every previous route scaled badly: the census is `~4k−2` nodes wide and the
per-instance decision costs `2^k` (NG-14, NG-16). Conjecture 13 has **no
dependence on k at all** — it is a poly-time *algorithm* (sort, test k+1
candidates) plus a proof obligation. It would deliver 1.3 on the 2-path grammar
outright, and it is the concrete answer to "an entirely different approach is
needed".

### Simultaneity: the clean route is BLOCKED, and what replaces it

Attempt (session 6). For each arc a let `G_a = { j : |L_a(j)| ≤ d_max }`.
Simultaneity is exactly `⋂_a G_a ≠ ∅`. If every `G_a` were a **contiguous run**
of j's, one-dimensional Helly would reduce the whole conjecture to *pairwise*
intersection — a clean proof.

**It is not.** Measured over 2,714 random instances: **167 arcs had a
non-contiguous `G_a`** (two separate blocks). So 1-D Helly does not apply and
that route is dead. Recorded rather than quietly abandoned.

Two facts survive the test and shape the next attempt:

1. **No arc ever had an empty `G_a`.** Every arc is individually satisfiable by
   some staircase — consistent with the single-row argument above (a monotone row
   crosses zero in steps ≤ d_max). So the entire difficulty is *simultaneity*,
   not any individual arc.
2. ~~**`G_a` never had more than 2 blocks.**~~ **RETRACTED within the same
   session** — that was a small-k artifact. Measured over 1,500 instances per k:
   max blocks = **2 (k=4), 3 (k=6), 3 (k=8), 4 (k=10), 4 (k=12)** — it *grows*
   with k. So the family is not one of 2-intervals, the Helly number for
   d-intervals grows with d, and **the bounded-local-check route dies as well**.
   Same over-reading as NG-12 (a trend inferred from limited coverage); caught in
   minutes here only because the speculation was tested immediately after being
   written down.

**Both obvious routes are therefore closed**, each with a concrete disproof:
contiguity fails (167/2,714 non-contiguous), and bounded interval-complexity
fails (blocks grow with k). Recorded so neither is re-attempted.

### ~~The structure a proof must use: laminarity~~ (moot for Conjecture 13 — see refutation)

Retained only as a note for any *future* rounding-family conjecture: the arcs'
subtree sets `S_a` form a **laminar** family, which every route tried here
discarded by treating rows as arbitrary ±1 patterns. If a different bounded
candidate family is proposed, laminarity is the unused ingredient.

### ORIGINAL (now moot) laminarity plan

What has *not* been used is the strongest fact available. The arcs' subtree sets
`S_a` are **subtrees of a tree, hence a LAMINAR family** — any two are nested or
disjoint — and `C[a,i] = [u_i ∈ S_a] − [v_i ∈ S_a]`. Every route tried so far
treated the rows as an arbitrary collection of ±1 patterns on a line, which
throws laminarity away; and an arbitrary ±1 family is exactly where the
counterexamples to contiguity and bounded-block-count live.

A proof should instead induct on the laminar tree: process arcs from the leaves
of the laminar order inward, maintaining an invariant on the partial loads, using
the fact that a child subtree's chord set is a *subset* of its parent's. That is
the one substantial unused ingredient, and it is also why the empirical facts hold
so robustly while the topological arguments fail.

Robust across all k tested (4..12, 7,500 instances): **no arc ever had an empty
`G_a`** — every arc is individually satisfiable. The whole content is
simultaneity, and laminarity is the tool that has not been tried.

**Discipline note:** before investing further in a proof, an adversarial search
for a *counterexample* to Conjecture 13 is running at k = 4..8 — reachable
precisely because the staircase objective has only k+1 disjunctions rather than
2^k. If the conjecture is false, that is the cheapest way to find out, and this
campaign has now been wrong three times in the optimistic direction.

### Relation to the literature

This is recognisably the sorted-prefix technique used for interval and
consecutive-ones discrepancy bounds, and plausibly close to Doerr's own route
(his EJC 2000 proof partitions columns by weight, which a sort implements). So
Conjecture 13 may be known for the unit box; the content here is that it appears
to survive **column scaling** — exactly the gap Q7′ (NG-18) isolates. Next
session: prove the simultaneity step, and check the sorted-prefix literature for
the scaled case.

---

---

## NG-20 (session 6): priority check — the spider IS prior art (Morell–Skutella 2022, Fig. 3)
**Status: NOVELTY CLAIM RETRACTED for Theorem 11. The caveat filed in NG-17 was
correct. What survives is narrower and better-located.**

### The prior art, verbatim

Morell & Skutella, *Math. Programming* 192 (2022) 477–496, §3 opening:

> "Notice that the lower bound (3) in Theorem 3 is tight in the following sense:
> For each ε > 0, there exists a digraph D together with a fractional flow x such
> that no unsplittable flow y satisfies y_a ≥ x_a − d_max + ε, for all a ∈ A;
> see the instance depicted in Fig. 3 with k = ⌈1/ε⌉."

Their Fig. 3 caption:

> "An instance with k demands of value 1. Flow x is given as follows: solid arcs
> have flow value k/(k+1) and dashed arcs carry flow value 1/(k+1). Notice that
> any unsplittable flow sends zero flow on some solid arc."

That is our spider: k unit demands, a shared hub, a private arc per terminal, and
the **same k/(k+1) weights** — with the long/short roles swapped relative to our
write-up, and the identical pigeonhole ("some solid arc gets zero"). Our
Theorem 11 is an independent rediscovery, nothing more.

Traub–Vargas Koch–Zenklusen (arXiv:2308.02651, §6) then make the consequence
explicit, citing both sides:

> "the capacity violation d_max of Theorem 1.7 … is tight. This is the case even
> if there is only a single demand, as observed already in earlier work
> [DGG99; MS22]. … **[MS22] gives a similar example showing that the lower bound
> on the unsplittable flow in Conjecture 1.5 is tight, even in planar graphs.**"

(TVZ's "Conjecture 1.5" is the uncosted two-sided Morell–Skutella statement —
our target.) So "1.3's constant cannot be improved to (1−ε)·d_max, even on
series-parallel/planar instances" is **published and cited**. NG-17's flow-facing
headline is retracted as a novelty claim; it stands only as an independently
verified restatement.

Also: MS22's own Fig. 3 gadget is series-parallel, so it doubles as the standing
tightness certificate for MSW25's series-parallel theorem — which MSW25 does not
state explicitly, but which follows immediately.

### What survives, and it is better located

1. **Q7′ — the column-scaled bound — is genuinely open.** Two independent
   searches found it posed nowhere, under any phrasing (weighted linear
   discrepancy, lattice approximation with unequal steps, `lindisc(AD)`, box
   discrepancy).
2. **The connection itself appears un-drawn.** Full-text greps of TVZ, Swamy
   (arXiv:2510.21287) and MSW25 find **no mention of Doerr, "linear discrepancy",
   or "totally unimodular" anywhere**. TVZ's discrepancy link is to the *prefix
   Beck–Fiala* problem, and they explicitly note "our notion of prefixes is
   different". So the observation that **MS Conjecture 1.3 restricted to 2-path
   instances *is* the column-scaled Doerr question** does not appear in either
   literature. That reduction — not the tightness gadget — is the campaign's most
   plausible contribution.
3. **The k = 3 column-scaled theorem** (NG-13/NG-18) remains the one proved
   instance of that open question.
4. **Conjecture 13** (NG-19, staircase suffices) remains the constructive route,
   with a partial proof.

### Loose end

`[1…1; I_k]` provably satisfies Doerr's published extremal characterisation
("n+1 rows, every n independent") and its lindisc is exactly `1 − 1/(k+1)`, so it
*is* an extremal example — but whether Doerr's paper displays this matrix could
not be confirmed (Combinatorica paywalled, no arXiv mirror, and the open-access
EJC 2000 predecessor covers only ≤2-nonzeros-per-row matrices, which excludes it).
Needs a library fetch of `10.1007/s00493-004-0007-x` to close.

### Method note

This is the third time in this campaign that the *first* framing of a result was
too strong and a targeted check corrected it (NG-7's false positive, NG-15's
refutation of NG-12, and now NG-20). In all three the correction came from
deliberately hunting for the disconfirming thing — an exact re-verification, a
wider stratum, a priority search. The pattern is worth keeping: **state the
novelty caveat when you feel the result is too pretty, then actually go check
it.** Here the caveat was written into NG-17 before the check was run, which is
why the retraction costs nothing.

---

---

### REFUTATION of Conjecture 13 (same session, before any proof was attempted)

The staircase objective has only **k+1 disjunctions** instead of 2^k, so it can be
*maximized* by MILP at k values where the full problem is hopeless. Doing that:

| k | max staircase-best deviation found (d_max = 1) |
|---|---|
| 4–7 | ≤ 1 |
| **8** | **1.636** (also 1.500, 1.345, 1.118, 1.040 on other structures) |

So **the k+1 sorted candidates do NOT always suffice** — at k = 8 the best
staircase can miss the d_max bound by 64%. Conjecture 13 is false.

**This does NOT refute Conjecture 1.3.** It says only that the staircase
*heuristic* is inadequate; the optimum over all 2^k roundings may still be within
d_max. A follow-up run recomputes the true 2^k optimum on every staircase-failing
instance to settle which. (If the true optimum ever exceeds d_max, that IS a
counterexample to 1.3.)

**Follow-up (decisive for direction): staircase-hard ≠ optimum-hard.** COMPLETED
run, all staircase failures at k = 5 and k = 8 checked against the true optimum
over all 2^k roundings: **zero counterexamples to Conjecture 1.3, and not one
close call.** Staircase values 1.04–1.64·d_max against true optima of
**1/3, 8/21, 323/840, 67/168, 2/5, 29/70, 3/7, 1/2** — i.e. 0.33–0.50·d_max,
while the extremal spider at k = 8 reaches 8/9 ≈ 0.89. The heuristic's failure
mode is **anti-correlated with actual difficulty**. So the regions where the heuristic fails are regions
where the real problem is *easy*, and they are **worthless as seeds for a
Conjecture 1.3 counterexample hunt**. This closes the last cheap idea for
finding a counterexample at k ≥ 7, which is where the census could not reach.

**Method note — the pattern, now four times over.** NG-12 (R = 3/4 "everywhere",
refuted at k = 4), the 2-interval claim (refuted at k = 6..12 within minutes),
and now Conjecture 13 (refuted at k = 8) all share one shape: **a clean pattern
observed at k ≤ 6 that fails at larger k.** In this campaign, k ≤ 6 evidence has
a track record of being actively misleading, because the interesting structures
need more chords than that to exist. The countermeasure that worked here was
ordering the work correctly: *refute first, prove second*. The refutation cost one
MILP sweep; the proof attempt it pre-empted would have cost a session, aimed at a
false statement.

---

*(entries below this line are appended by search sessions)*
