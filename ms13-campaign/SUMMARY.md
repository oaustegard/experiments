# Two-sided unsplittable-flow rounding on out-trees: a reduction to column-scaled discrepancy of network matrices

**Report on a computational campaign against Morell–Skutella Conjecture 1.3.**
oaustegard/claude-workspace, issue #169. Work performed 2026-07-24/25.
Supporting code, ledger and certificates: this directory; `NOGOS.md` is the
primary record and contains every claim below with its proof or its refutation.

---

## Abstract

Morell–Skutella conjecture that for every acyclic single-source unsplittable-flow
(SSUF) network and every fractional flow `x` there is an unsplittable routing `P`
with `|f^P(a) − x(a)| ≤ d_max` on every arc. We attempted to refute it. We did
not. Instead we (i) show that the conjecture restricted to instances with exactly
two `s→t_i` paths per terminal is *equivalent* to a linear-discrepancy question
about **network matrices with demand-scaled columns**, (ii) prove that question
for `k = 3` terminals, exactly and unconditionally, with the sharp constant
`3/4`, and (iii) record a ledger of refuted constructions, including two
conjectures of our own. The reduction in (i) appears not to have been drawn in
either the unsplittable-flow or the discrepancy literature; (ii) is, as far as we
can determine, the first non-trivial case of the column-scaled generalisation of
Doerr's theorem on the linear discrepancy of totally unimodular matrices. No
counterexample to Conjecture 1.3 was found, and we give arithmetic reasons why
enumerative search cannot decide it.

Numbering follows Swamy (arXiv:2510.21287): "Conjecture 1.3" is the two-sided,
cost-free, acyclic statement, i.e. Morell–Skutella's own "Conjecture 1". The
literature numbers this statement three different ways; see §7.

---

## 1. Setting and notation

An SSUF instance is a DAG `G = (V,A)`, source `s`, terminals `t_1..t_k` with
demands `d_i > 0`, `d_max = max_i d_i`, and a fractional flow `x ∈ Q_G`
decomposed into per-terminal path flows. A *routing* `P` assigns each terminal
entirely to one `s→t_i` path; `f^P(a)` is the induced arc load. Define

    t(x) = min over routings P of max over arcs a of |f^P(a) − x(a)|  −  d_max,

so `(G,d,x)` refutes Conjecture 1.3 iff `t(x) > 0`.

## 2. Normal form: 2-path instances are out-trees

**Definition.** An instance is *path-closed* if each terminal's declared path set
is all of its `s→t_i` graph paths. (Verification against a declared sub-menu is
unsound — see §6.)

**Lemma 4 (splice closure).** In a path-closed instance with exactly two paths
per terminal, sharing is prefix-only and paths never reconverge; consequently the
instance is an **arborescence rooted at `s` whose terminals are sinks of
in-degree 2**. Conversely every such "out-tree instance" is path-closed by
construction.

This is the formal content of the informal "splice closure" constraint used in
prior counterexample searches: a terminal's two paths may share a prefix, and any
reconvergence would create a third path.

## 3. The reduction

Let `T` be the tree, and for terminal `i` let `u_i, v_i` be its two attachment
nodes ("chords"). Write `z_i = 1` if `i` routes via `u_i`, and `w_i` for the
fractional weight on that path. For a tree arc `a` with head-subtree `S_a` put

    C[a,i] = [u_i ∈ S_a] − [v_i ∈ S_a] ∈ {−1,0,+1}.

**Lemma 5.** `f^P(a) − x(a) = Σ_i C[a,i]·d_i·(z_i − w_i)` for every tree arc `a`,
and terminal arcs have deviation `d_i w_i` or `d_i(1−w_i)`, hence never exceed
`d_max`. `C[a,i] ≠ 0` exactly when `a` lies on the tree path between `u_i` and
`v_i`, signed by side — so **`C` is the network matrix of `T` with respect to the
terminal chords, and is therefore totally unimodular**.

Consequently, with `lindisc(A) = max_{w ∈ [0,1]^k} min_{z ∈ {0,1}^k} ‖A(z−w)‖_∞`,

> **Conjecture 1.3 restricted to 2-path instances ⟺
> `lindisc(C·diag(d)) ≤ d_max` for all network matrices `C` and demands `d`.**

Two immediate consequences.

**Lemma 6 (equal demands).** `[C; I]` is TU, so by Hoffman–Kruskal integrality
the polyhedron `{ζ : ⌊Cw⌋ ≤ Cζ ≤ ⌈Cw⌉, 0 ≤ ζ ≤ 1}` is integral and contains `w`;
any integral point in it gives `|C(z−w)|_∞ < 1`. Hence equal demands never
refute Conjecture 1.3. (Doerr's theorem, §4, sharpens this to `1 − 1/(k+1)`.)

**Lemma 7 (cycle elimination).** If the chords with fractional `ζ_i` contain a
cycle in the multigraph `H` on tree nodes (one edge `{u_i,v_i}` per chord), the
signed combination of their columns vanishes identically, so `ξ` may be shifted
around the cycle **changing no arc load**. Hence WLOG the fractional chords form
a forest; at most `n−1` are fractional at a critical point, independently of `k`.

**Lemma 10 (forest reduction).** Iterating Lemma 7 from `ζ = w` (all loads zero)
gives `R(C) ≤ max over forest sub-multisets F ⊆ C of R(F)`, where `R` denotes the
normalised worst-case ratio of §4. Every non-forest chord set therefore reduces
to strictly fewer chords.

## 4. The scaled discrepancy question, and the main theorem

Normalise `d_max = 1` and define, for a tree-plus-chords structure,

    R(structure) = sup over demands d and w of
                   [min over roundings of max-arc deviation] / d_max.

`R` depends on the structure only through its **row-set** `{C[a,·]}` modulo row
sign flips, column sign flips (chord reorientation, which also swaps
`p_i = d_i w_i ↔ q_i = d_i(1−w_i)`) and column permutation; and `R` is monotone
non-decreasing under adding rows. Conjecture 1.3 fails on a structure iff `R > 1`.

**Known (Doerr).** For general totally unimodular `A` with `n` columns,
`lindisc(A) ≤ 1 − 1/(n+1)`, sharply (Combinatorica 24 (2004) 117–125). Network
matrices are TU, so the **equal-demand** case of our question is settled with the
sharp constant. *Caveat: we could not obtain this paper's full text (paywalled,
no preprint); the statement and its extremal characterisation are corroborated by
consistent independent secondary sources only.*

**Open question Q7′.** Does column scaling exceed the unit-box bound? I.e. is
`lindisc(C·diag(d)) ≤ d_max·(1 − 1/(k+1))` for network matrices? Neither proved
nor refuted anywhere we could reach, under any of: weighted linear discrepancy,
box discrepancy, `lindisc(AD)`, lattice approximation with unequal steps, vector
balancing with unequal norms.

> **Theorem (main).** For every out-tree structure with `k = 3` chords,
> `R = 3/4` exactly. Equivalently `lindisc(C·diag(d)) = (3/4)·d_max` at the
> supremum over demands, for every network matrix with three chords — the `k = 3`
> case of Q7′, attained.

*Proof.* Two halves.

**Lower bound.** The minimal row-set attaining it has three rows, normalised
`{(1,1,1), (1,1,0), (1,0,1)}` (the Rybin-instance type). With column 1 the chord
present in every row, take `p_i = q_i = d_i/2`, `d = (t,1,1)`. If `δ_2, δ_3` take
opposite signs the pair rows force `t/2 + 1/2`; if equal signs, the best choice of
`δ_1` gives `1 − t/2`. Hence `min over roundings = min(t/2 + 1/2, 1 − t/2)`,
maximised where the branches cross, at `t = 1/2`, value `3/4`. ∎

**Upper bound.** `R ≤ v` iff the `2^k` polytopes
`P_z = {(p,q) : all |loads| ≤ v}` cover the normalised box. The negation is a
disjunctive system (per rounding, a strictly violated (row, side)), searched
depth-first with an exact rational LP per node and the margin maximised
symbolically. Exactly two maximal row-set classes exist; both were proved
`R ≤ 3/4` with zero surviving leaves (15,975 and 14,440 nodes; 191,700 and
173,280 exact LP calls). Monotonicity in the row-set extends the bound to all
classes. ∎

**Completeness of the class census.** A row is determined by which chord paths
contain an arc and with what sign; subdividing at an unmarked degree-2 node
duplicates a row rather than creating one, and per-row sign quotienting makes the
root irrelevant. So unmarked nodes may be assumed to have degree ≥ 3, and with
6 marked nodes there are at most 4 of them: **`n ≤ 10` realises every row-set**.
Exhaustive enumeration through `n = 10` yields **30 classes, with no new class
beyond `n = 7`** and exactly two maximal ones — the two proved above. The theorem
is therefore unconditional.

## 5. Negative results and refuted constructions

Recorded in `NOGOS.md` with the argument that killed each.

| # | Family / claim | Verdict |
|---|---|---|
| NG-1 | Ceilings-only parallel-chain pigeonhole, any `m ≥ 2` chains, `k = m+1` | Impossible: summation gives `(m−1)d_1 > m·d_max`. Corollary: floors `x(a) > d_max` are essential |
| NG-3 | Pure coverage-pigeonhole (more parallel floors than terminals can cover) | Unrealisable: antichain mass bound `Σ x(a) ≤ Σ d_i` |
| NG-4 | Cost-preserving violation constant `β*` via odd-hole/clique conflict ladders | Self-limiting at `β* = 7/6`; never approaches the 2 needed for the STVZ contrapositive |
| NG-6 | Minimal `k = 3` single-floor implication gadget | Impossible for every demand vector (blocking-cost summation) |
| NG-9 | Any out-tree whose tree+chords graph is K4-minor-free | Already proved by Majthoub Almoghrabi–Skutella–Warode (series-parallel). 85.3% of enumerated structures; `n ≤ 3` dead by vertex count |
| NG-14 | `k = 4` by the same census method | Compute no-go: bound widens to `n ≤ 14`, giving `1.24×10^10` configurations (~2,070 h) plus a `2^k`-deep search |
| **Conj. 12.1** | `R ≤ 3/4` for all `k` (**ours**) | **Refuted at `k = 4`**: an equal-demand instance attains `4/5` exactly |
| **Conj. 13** | The `k+1` sorted "staircase" roundings always suffice (**ours**) | **Refuted at `k = 8`**: staircase-best reaches `1.64·d_max` |

**Conjecture 12.2 (open).** `R_max(k) = k/(k+1)`, attained; hence
`sup_k R_max(k) = 1`. `R_max` is non-decreasing in `k` (zero demands degenerate a
`k`-chord instance to `k−1`), so the supremum is well defined and `≥ 4/5`.
`k = 3` is proved; `k = 4` gives `4/5`.

**Tightness (prior art).** The construction attaining `k/(k+1)` — `k` unit
demands, a shared hub arc, a private arc per terminal, `w = k/(k+1)` — is
Morell–Skutella's own Fig. 3, which they use to show their Theorem 3's lower
bound is tight; Traub–Vargas Koch–Zenklusen §6 cite it (with Dinitz–Garg–Goemans
for the other side) and extend it to planar graphs. We rediscovered it
independently; it is not new. It follows that `d_max` cannot be improved to
`(1−ε)d_max` on this class, and — since the gadget is series-parallel — that
MSW25's bound is asymptotically tight on the simplest possible instance.

**A negative that closed the last search idea.** The instances where the sorted
heuristic fails worst are the instances where the true optimum is *easiest*: at
`k = 8`, staircase values `1.04–1.64·d_max` against true optima of
`1/3 … 1/2·d_max`. The heuristic's failure mode is anti-correlated with
difficulty, so staircase failures are worthless as counterexample seeds.

## 6. Methodology, and one instructive failure

All decisions are exact (`fractions.Fraction`, exact rational LP); floating point
appears only in screening and never in a verdict. Calibration gates run before
any search (`tests/test_calibration.py`, 5/5): a hand-derived triangle instance;
agreement between the exact membership LP and an independently formulated
`scipy.linprog` check at 48 points; a hand-computed two-sided instance; a
malformed-path guard; and a **two-sided** check of the certified decision
procedure — it must *find* a bad `w` below a known optimum and *prove* none
above it.

That last gate exists because of a specific failure. An early candidate passed
abstract satisfiability, exhibited a consistent `x`, scored `t(x) = 1/8 > 0`, and
was confirmed by two independently written realisation code paths. It was wrong:
the realisation was not path-closed, the true routing space was 256 rather than
8, and 23 routings satisfied the bound (`t(x) = −3/8`). Both "independent" paths
shared one blind spot — verification against a declared path menu — so their
agreement was systemic rather than confirmatory. `check_path_closure` is now a
mandatory gate, and the reconstructed Rybin instance passes it.

Separately, a first formulation of the certified decision MILP had inverted big-M
polarity and a sign error on the margin; either would have reported "no
counterexample" everywhere, and a one-sided test would have passed both.

**Search-quality note.** Every over-claim in this campaign had the same shape: a
clean pattern at `k ≤ 6` that fails at larger `k` (Conj. 12.1 at `k = 4`; a
"`G_a` is a union of ≤ 2 intervals" claim at `k = 6..12`; Conj. 13 at `k = 8`).
Small-`k` evidence here is not merely weak but actively misleading, because the
structures that matter need more chords to exist. The procedure that worked was
to **attempt refutation before investing in proof**: refuting Conjecture 13 cost
one MILP sweep and pre-empted a proof effort aimed at a false statement.

## 7. A numbering hazard

The same statement is numbered three ways: Morell–Skutella's "Conjecture 1"
(their Conjecture 2 adds costs); Swamy arXiv:2510.21287's "Conjecture 1.3"
(1.4 with costs; 1.5 the `2d_max` cost version); and Traub–Vargas Koch–Zenklusen
arXiv:2308.02651's "Conjecture 1.5" (their 1.3 is Goemans'). Citing
"Conjecture 1.3" without naming the source is ambiguous.

## 8. Open problems

1. **Q7′** — `lindisc(C·diag(d)) ≤ d_max(1 − 1/(k+1))` for network matrices.
   Proved for `k = 3`; equal demands known for all `k`. Suggested attack: Doerr's
   own combinatorial method (partition columns by weight into extreme and
   moderate classes via Ghouila-Houri, round the extremes and balance the
   moderates by hereditary discrepancy ≤ 1), which reasons about individual
   column weights against a partition of `[0,1]` and so admits a `d_i`-dependent
   partition. Lemmas 7 and 10 permit assuming the chords form a forest.
   Laminarity of the subtree family `{S_a}` is the strongest unused hypothesis;
   every route tried here discarded it by treating rows as arbitrary ±1 patterns
   on a line, which is exactly where the counterexamples to those routes live.
2. **Conjecture 12.2** — the rate `k/(k+1)`. `k = 4`'s upper bound is unproved.
3. **`k ≥ 3` paths per terminal.** Lemma 4's normal form is specific to two
   paths; the general case has richer closure structure and is untouched.
4. **Verify Doerr's constants from primary source** (`10.1007/s00493-004-0007-x`).

## References

- S. Morell, M. Skutella. Single source unsplittable flows with arc-wise lower
  and upper bounds. *Math. Programming* 192 (2022) 477–496.
- C. Swamy, V. Traub, L. Vargas Koch, R. Zenklusen. Unsplittable Cost Flows from
  Unweighted Error-Bounded Variants. arXiv:2510.21287.
- V. Traub, L. Vargas Koch, R. Zenklusen. Single-Source Unsplittable Flows in
  Planar (and Bounded-Genus) Graphs. arXiv:2308.02651; *Math. Prog.* 2026.
- M. Majthoub Almoghrabi, M. Skutella, P. Warode. Integer and Unsplittable
  Multiflows in Series-Parallel Digraphs. arXiv:2412.05182; IPCO 2025.
- Y. Dinitz, N. Garg, M. Goemans. On the single-source unsplittable flow problem.
  *Combinatorica* 19 (1999).
- B. Doerr. Linear Discrepancy of Totally Unimodular Matrices. *Combinatorica* 24
  (2004) 117–125. (Predecessor: Linear Discrepancy of Basic Totally Unimodular
  Matrices, *EJC* 7 (2000) #R48 — restricted to ≤ 2 nonzeros per row.)
- L. Lovász, J. Spencer, K. Vesztergombi. Discrepancy of set-systems and
  matrices. *Europ. J. Combin.* 7 (1986) 151–160.
- D. Rybin (2026-07-21), announcement of a counterexample to the
  Dinitz–Garg–Goemans cost conjecture. Social-media only; instance reconstructed
  here from its published invariants (`rybin.py`).

## Reproduction

```bash
cd experiments/ms13-campaign
python3 tests/test_calibration.py   # 5 gates
python3 test_rybin.py               # 13 anchor assertions
python3 k3_proof.py                 # k=3 census + exact branch-and-bound
python3 fast_census.py              # census closure through m=10
python3 ladder.py                   # beta* mechanism ladder
```
