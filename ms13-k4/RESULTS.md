# ms13-k4: Q7′ and Conjecture 12.2 at k = 4

Session 2026-09-01 (Claude Code on the Web, hub `claude-workspace`), reopening the
Morell–Skutella 1.3 campaign (`ms13-campaign/`, claude-workspace#169, closed 2026-07-25 as
"the remaining mathematics is not compute-bound").

## Result

**Theorem (k = 4).** For every out-tree structure with four chords,
`R = 4/5` exactly. Equivalently `lindisc(C·diag(d)) ≤ (4/5)·d_max` for every network matrix
`C` with four chords and every demand vector `d`, and the bound is attained.

This is Conjecture 12.2 (`R_max(k) = k/(k+1)`) at `k = 4`, and the `k = 4` case of Q7′
(column scaling does not exceed Doerr's unit-box bound `1 − 1/(k+1)`). The campaign had
proved `k = 3` and left `k = 4` open because its structure census was sized at ~2,070 CPU
hours (NG-14). Both halves here are exact: the upper bound is a branch-and-bound over an
exact rational LP with zero surviving leaves per type, the lower bound is a rational witness
re-evaluated by a separately written brute-force evaluator (`verify_witness.py`, no shared
code with the prover).

A second fact came out of the run: **all 14 maximal four-chord types have `R = 4/5`**, and
every one attains it at *unit* demands, mostly at `w = (1/5, 2/5, 3/5, 4/5)` up to the
symmetry group. Unequal demands never improve on the unit-demand extremum at `k = 4`, which
is the content of Q7′ in the strongest per-type form.

The unit-demand half of that is not a coincidence. Doerr (Combinatorica 24, 2004)
characterises the TU matrices with `n` columns attaining `1 − 1/(n+1)` as those containing
`n+1` rows every `n` of which are linearly independent. `doerr_extremal.py` finds such a
`(k+1)`-row subset in all 14 maximal types at `k = 4` and in both at `k = 3` (the two leaf
rows `e_i` plus internal splits do it; for the type whose chords include a cherry, a
different five rows). So every maximal type is Doerr-tight at unit demands by his theorem,
and the new content at `k = 4` is only the other half: no demand vector pushes any type past
`4/5`. The `k ≥ 5` version of the unit-demand half is a question about binary-leaf pairings
containing such a row subset; the `k ≥ 5` version of the other half is Q7′.

| type | rows | lower bound (verified) | at unit demands | upper bound proved ≤ 4/5 | nodes | exact LPs | s |
|---:|---:|---:|:-:|:-:|---:|---:|---:|
| 0 | 9 | 4/5 | yes | yes | 1,710 | 30,780 | 10 |
| 1 | 9 | 4/5 | yes | yes | 2,985 | 53,730 | 21 |
| 2 | 9 | 4/5 | yes | yes | 909 | 16,362 | 5 |
| 3 | 7 | 4/5 | yes | yes | 629 | 8,806 | 3 |
| 4 | 9 | 4/5 | yes | yes | 1,343 | 24,174 | 8 |
| 5 | 9 | 4/5 | yes | yes | 2,704 | 48,672 | 18 |
| 6 | 9 | 4/5 | yes | yes | 3,241 | 58,338 | 23 |
| 7 | 9 | 4/5 | yes | yes | 3,856 | 69,408 | 28 |
| 8 | 9 | 4/5 | yes | yes | 7,454 | 134,172 | 64 |
| 9 | 8 | 4/5 | yes | yes | 5,259 | 84,144 | 40 |
| 10 | 9 | 4/5 | yes | yes | 7,414 | 133,452 | 62 |
| 11 | 9 | 4/5 | yes | yes | 5,650 | 101,700 | 40 |
| 12 | 9 | 4/5 | yes | yes | 6,730 | 121,140 | 56 |
| 13 | 9 | 4/5 | yes | yes | 3,111 | 55,998 | 22 |

Type indices refer to `types2_k4.json`; per-type data is in `k4_summary.json`. Total prover
time for the upper bound: about 400 s on one core.

## Maximal types as Buneman split systems

A row of the network matrix is a tree arc `a`, with `C[a,i] = [u_i ∈ S_a] − [v_i ∈ S_a]`
where `S_a` is the far side of the arc and `u_i, v_i` are the two attachment nodes of chord
`i`. So a row is a function of the *split* of the `2k` chord endpoints induced by the arc; the
row's sign is the choice of side, which the objective `|·|` already quotients out. A set of
splits of a labelled set is realized by a tree if and only if the splits are pairwise
compatible (Buneman), and every such tree refines to a binary tree with all `2k` labels at
leaves, whose split system contains the original one (moving a label from an internal node
to a new pendant leaf preserves every existing split and adds a trivial one). `R` is monotone
non-decreasing under adding rows (campaign Lemma, K3_PROOF.md). Therefore:

> the maximal row-set types with `k` chords are exactly the split systems of binary trees on
> `2k` leaves whose leaves are paired into `k` chords, modulo the group.

At `k = 4` that is 10,395 labelled binary trees, or 4 tree shapes × 105 leaf pairings,
enumerated in under a minute (`splits.py`, `splits2.py`, both agree). At `k = 3` the same
enumeration returns the campaign's two maximal 6-row types exactly
(`k3_census_complete.json`), which is the calibration; and the campaign's `k = 4` refuting
row-set (NG-15) is contained in type 6, as it must be. The campaign's census enumerated
trees on up to 14 nodes with 8 markers because it allowed markers at internal nodes; the
refinement argument shows those never produce a row-set outside the binary-leaf family.

## Fail-first branching in the exact prover

`k3_proof.exact_upper_bound_bb` walks the 2^k roundings in lexicographic order and, for
each, branches over the `(row, side)` options that can still be strictly violated. At
`k = 4` (16 roundings, 18 options) the first type had not finished after 50,000 nodes and
900,000 exact LPs. `bb2_k4.py` / `bbk.py` keep the same disjunctive system, the same exact
`Fraction` simplex, and the same leaf certificate, and change only the branching order:
the next rounding is the one with the fewest `(row, side)` options that hold at the parent
LP's margin-maximizing point. Ordering cannot change the outcome of the search, only the node count. The
node counts above are the result; the same code re-proves the campaign's `k = 3` theorem in
82 and 122 nodes (campaign: 15,975 and 14,440).

Two-sided calibration, per campaign rule (NG-7, GATE 5): at `v = 3/4` the prover must
*find* a counterexample on every type that attains 4/5. It does, on all 14, each time
returning a unit-demand point with margin exactly `1/20 = 4/5 − 3/4`
(`bb2_k4_3_4_*.json`). Those points are the lower-bound witnesses in the table.

## A hand lemma for the spider

On the spider type `{(1,…,1), e_1, …, e_k}` (hub arc plus one private arc per chord; the
Morell–Skutella Fig. 3 gadget) the bound `R ≤ k/(k+1)` holds for *every* demand vector, not
only unit demands. Write `v = k/(k+1)`; the private rows force any chord with `p_i > v` up
(then `q_i < 1/(k+1)`) and any chord with `q_i > v` down. The remaining chords are free on
their private arcs. Flipping a free chord moves the hub load by `p_i + q_i ≤ 1`; the
all-down hub load is at most `Σ_{forced up} q_i < k/(k+1) = v`, the all-up load is at least
`−v`, so the chain of one-flip moves crosses into `[−v, v]` because that interval has length
`2v ≥ 1`. So the one type known to be Doerr-tight cannot be pushed past Doerr by column
scaling; at `k = 4` the computation says the same of the other thirteen.

## The unit bound via Hoffman–Kruskal and Carathéodory

The unit-demand bound `1 − 1/(k+1)` for a TU matrix has a short proof that is worth
recording because it says exactly what Q7′ is missing. Let `P = {x ∈ [0,1]^k : ⌊Cw⌋ ≤ Cx
≤ ⌈Cw⌉}`. It contains `w` and is integral (Hoffman–Kruskal), so `w` is a convex combination
of at most `k+1` of its integer vertices `x^0, …, x^k` (Carathéodory), each with per-row
error in `{−f_a, 1 − f_a}` where `f_a` is the fractional part of `(Cw)_a`. The errors of the
vertices average to zero on every row, so the vertices that are "high" on a row with `f_a <
1/(k+1)` carry total weight `f_a < 1/(k+1)`, and likewise for "low" on rows with `f_a >
k/(k+1)`. The vertex of largest weight has weight at least `1/(k+1)` and so belongs to no
bad set: all its errors lie in `[−k/(k+1), k/(k+1)]`.

With scaled columns the averaging step survives unchanged, but the two-valued error
structure does not: it needs `Cx` integral for integral `x`, and `C·diag(d)` has no such
property. That is the precise gap. A proof of Q7′ needs a substitute for "each row's error
takes two values one unit apart", and nothing in the campaign's routes (Doerr's
extreme/moderate partition, Ghouila-Houri, laminarity) supplies it yet.

## Prior art

Two web searches (2026-09-01) for weighted or column-scaled linear discrepancy of TU or
network matrices, and for rounding within the maximum column weight, found Doerr 2000
(EJC, basic TU matrices) and Doerr 2004 (Combinatorica) on the unit case, the Ghouila-Houri
partition theorem, and randomized-rounding heuristics for unsplittable flow. Neither
search found a statement of Q7′ or a weighted analogue of Doerr's bound; two searches is
not a survey. The campaign's own literature gate (phase 3, NG-9) reached the same absence
by full-text grep of TVZ, Swamy et al. and MSW25.

## Open problems

1. **Q7′ / Conjecture 12.2 for `k ≥ 5`.** The enumerator handles `k = 5` (11 shapes × 945
   pairings) and the prover is expected to scale to 32 roundings; see `types2_k5.json` and
   the `bb2_k5_*` files if present.
2. **A proof.** The gap is stated above. The empirical regularity that *every* maximal type
   attains `k/(k+1)` at unit demands (both at `k = 3` and `k = 4`) suggests a lemma about
   binary-leaf split systems rather than about any one gadget.
3. **Conjecture 1.3 itself** (bound `d_max`, all `k`) is untouched by this; `k/(k+1) < 1`
   for the proved cases only.

## Files

- `splits.py`: labelled-binary-tree enumeration, canonical row-set under the group.
- `splits2.py`: shape × pairing enumeration (same output, ~100× cheaper); `types2_k{3,4}.json`.
- `kproof.py`: the campaign's `k3_proof.py` with `K` taken from `MS13_K` (unchanged otherwise).
- `bbk.py`: fail-first exact B&B. `MS13_K=4 python3 bbk.py 4/5 0,1,…,13 0 types2_k4.json`.
- `verify_witness.py`: independent brute-force evaluator for a `(rows, p, q)` witness.
- `doerr_extremal.py`: Doerr's `(k+1)`-rows-every-`k`-independent test per type.
- `k4_summary.json`, `k4_bb2_4_5_*.json`, `bb2_k4_3_4_*.json`: the certificates.
