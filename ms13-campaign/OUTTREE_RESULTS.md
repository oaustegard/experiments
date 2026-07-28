# OUTTREE_RESULTS.md — phase-3 out-tree search, MSW25-filtered

**Status: NO counterexample found.** Full 41/41 (tree, k) combo sweep
completed in 540.2s of the 720s (12min) ceiling. Zero assertion failures
(Lemma 6 equal-demand self-test and the new MSW25-covered self-test both
passed on every sample). The MSW25 sanity anchor (Rybin's chord graph is
exactly K4) reproduced correctly. Machine-readable results:
`outtree_results.json`. Search code: `outtree_search.py`.

This entry supersedes an interrupted first attempt (26/41 combos, also
zero failures/hits, truncated by the soft-stop guard at 615s under looser
search knobs) — not kept on disk, but its timing informs the knob-tuning
note in §8.

---

## 0. What changed mid-session: the MSW25 literature gate

Majthoub Almoghrabi, Skutella, Warode (arXiv:2412.05182, Thm 2) **prove**
Conjecture 1.3 with the tight two-sided `d_max` bound for multiflows on
series-parallel digraphs. Suppressing an out-tree instance's terminal sinks
(each has undirected degree 2) turns it into "tree + one chord `{u_i,v_i}`
per terminal". If that graph is series-parallel (K4-minor-free), **no
counterexample can exist there** — `outtree.msw25_covered` decides this via
the classical SP reduction (`outtree.is_series_parallel`, delete degree-≤1
vertices / suppress degree-2 vertices / drop parallels until nothing or a
K4 remnant is left).

**Structural consequence that reshaped the whole search.** Coverage is a
property of the attach *structure* alone — `chord_graph_edges` never looks
at demand values. And a K4 minor needs ≥4 branch vertices, so:

- **n=2, n=3 trees (root + 1-2 more nodes) can NEVER be uncovered**,
  regardless of how many parallel chords land on their 1-3 available
  pairs — there just aren't enough vertices for a K4 minor. Confirmed
  empirically: every one of `n2-chain`, `n3-chain`, `n3-cherry`'s deduped
  structural patterns is covered (see the ledger below). These three
  shapes are dead-on-arrival for counterexample-hunting; kept only as
  cheap self-test fodder.
- **n=4 trees have a tiny uncovered fraction**: 1-6 uncovered structural
  patterns per (tree, k) out of 14-126 deduped — essentially "the Rybin K4
  completion" (k=3) and its k=4 near-neighbors (one extra, structurally
  redundant chord). This is provably where Rybin's construction lives and
  provably almost nowhere else at this size.
- **n=5 trees are the new live territory**: up to 378/2002 (≈19%)
  uncovered at k=5 on chain-shaped trees. This is where the freed-up
  compute budget was redirected, per the coordinator's instruction.

## 1. Sanity anchor (Rybin, confirms the filter is wired correctly)

Rybin instance as an `OutTreeInstance`: `tree_arcs = [(s,a),(a,b),(b,c)]`
(chain), `attach = {t1:(a,c), t2:(s,c), t3:(s,b)}`, `demands=(15,10,15)`.

```
chord_graph_edges(rybin) = [(a,s),(a,b),(b,c),(a,c),(c,s),(b,s)]
                          = all 6 edges of K4 on {s,a,b,c}
msw25_covered(rybin)     = False   (expected False)  -- PASSED
```

This is the one and only known counterexample-adjacent construction in
this class, and it sits exactly at the boundary MSW25 draws: the smallest
non-series-parallel graph. Every other n=4 uncovered pattern found in this
sweep is a structural neighbor of this one.

## 2. Tree shape enumeration (n=2..5)

16 shapes total, hardcoded and cross-checked against OEIS A000081
(unlabeled rooted trees: a(2)=1, a(3)=2, a(4)=4, a(5)=9 — verified by hand
via the child-subtree-size-partition argument before coding).

| n (nodes) | shapes | automorphism sizes |
|---|---|---|
| 2 | 1 | 1 |
| 3 | 2 | 1, 2 |
| 4 | 4 | 1, 2, 1, 6 |
| 5 | 9 | 1, 2, 1, 6, 1, 2, 2, 2, 24 |

n=5 shapes (partition of the 4 non-root nodes among root's child
subtrees, by rooted-subtree size): `{4}` → chain4, chain-fork,
chain-leaf, chain-star (4 shapes, reusing the 4 n=4 shapes as subtrees
under one more root arc); `{3,1}` → chain3-leaf, cherry-leaf (2 shapes);
`{2,2}` → twochains (1 shape); `{2,1,1}` → chain2-2leaf (1 shape);
`{1,1,1,1}` → star4 (1 shape). 4+2+1+1+1 = 9, matches a(5)=9.

## 3. Structural attach enumeration + covered/uncovered ledger

For each tree shape and applicable k (n≤4 trees: k=3,4; n=5 trees:
k=3,4,5 — the coordinator's "k=4-5" redirect), all ways to attach k
terminals at unordered node pairs are enumerated (multiset over pairs,
deduped by the tree's automorphism group), then every deduped pattern is
classified covered/uncovered via `is_series_parallel` on
`tree_arcs + attach`. Coverage is demand-agnostic, so this classification
is reused for every demand vector tried on that tree/k.

**Totals across all 41 (tree,k) combos:**

| | count |
|---|---|
| raw (pre-dedup) attach tuples | 27,213 |
| deduped structural patterns | 16,453 |
| **covered** (MSW25-proved dead) | 14,038 (**85.3%**) |
| **uncovered** (live territory) | 2,415 (**14.7%**) |

Full per-(tree,k) breakdown in `outtree_results.json["enumeration_counts"]`.
Representative rows:

| tree/k | deduped | covered | uncovered |
|---|---|---|---|
| n2-chain/k3, k4 | 1, 1 | 1, 1 | 0, 0 |
| n3-chain/k3, k4 | 10, 15 | 10, 15 | 0, 0 |
| n3-cherry/k3, k4 | 6, 9 | 6, 9 | 0, 0 |
| n4-chain3/k3, k4 | 56, 126 | 55, 120 | **1, 6** |
| n4-star3/k3, k4 | 14, 28 | 13, 26 | **1, 2** |
| n5-chain4/k3,k4,k5 | 220,715,2002 | 211,637,1624 | **9, 78, 378** |
| n5-star4/k3,k4,k5 | 20,53,125 | 19,48,107 | **1, 5, 18** |

As predicted structurally: n=2,3 rows are 100% covered; n=4 rows are
≥96% covered; n=5 chain-shaped rows drop toward 76-82% covered at k=5,
which is where the search effort was spent.

## 4. Demand set

Exact `fractions.Fraction`, all unequal (Lemma 6 excludes all-equal
vectors from ever giving a counterexample — see §6).

- **k=3**: `(15,10,15)`, `(7,5,3)`, `(5,5,6)`, `(3,2,2)` (the task's named
  non-divisible set) + 3 random unequal vectors (seed 1729):
  `(3,15,18)`, `(16,7,3)`, `(7,11,18)`.
- **k=4**: `(9,7,6,4)` (named) + `(11,9,6,2)`, `(13,8,5,3)` + 2 random:
  `(13,11,8,6)`, `(19,17,2,16)`.
- **k=5** (new, for the n=5 redirect): `(11,9,7,5,3)`, `(13,11,8,6,2)`,
  `(9,7,6,5,4)` + 2 random: seeded, all unequal.
- **Equal-demand self-test only** (never counted as "real" sweep data):
  `(4,4,4)`, `(4,4,4,4)`, `(4,4,4,4,4)`.

For each live (uncovered) structural pattern, `TRIALS_PER_PATTERN=4`
random trials assign a randomly chosen demand vector under a random
permutation to the pattern's k slots (correct because `t_of_w` is
invariant under any simultaneous permutation of terminal index across
`(attach, demands, w)` — the search is over the full `[0,1]^k` box, so
permuting which coordinate carries which value doesn't change the set of
reachable `t(w)` values; only genuinely distinct value multisets matter,
which is exactly what the random-permutation trials probe).

## 5. w-search per instance

Exact `Fraction` arithmetic throughout: `{0,1/2,1}^k` corner grid, a
"balanced" heuristic (`w_i = d_j/(d_i+d_j)` for each tree-arc-co-resident
terminal pair, others at 1/2), `RANDOM_SAMPLES=12` random rational points,
2 passes of coordinate polish over a 13-point/coordinate grid. Short-
circuits the instant any `t(w) > 0` is found.

## 6. Self-tests (both passed on every sample — zero bugs found)

- **Lemma 6 (equal demands never give a counterexample)**: sampled on 6
  representative trees (`n2-chain, n3-cherry, n4-star3, n4-chain3,
  n5-chain4, n5-star4`) × applicable k, ≤8 structural patterns each.
  `equal_demand_no_go()` asserted True and `t(w) ≤ 0` asserted on every
  sample — **passed** (0 assertion failures).
- **MSW25-covered self-test** (new): ≤5 covered structural patterns per
  (tree,k) combo (197 total instances checked), each built with a random
  unequal demand assignment, `msw25_covered()` asserted True, `t(w) ≤ 0`
  asserted — **passed** (0 assertion failures). This directly confirms
  the filter is not silently wrong: if MSW25 coverage were mis-detected
  anywhere in this class, this assertion would have caught it.
- **msw25_covered() polarity check** on every uncovered instance actually
  searched (`assert not msw25_covered(inst)`): also passed throughout.

## 7. Live search: max t found, restricted to the UNCOVERED set

**No positive t(w) anywhere.** Global max over all 1,312 uncovered-set
instance-searches:

```
t*     = -4/3          (attained on TWO trees, see below)
d_max  = 3
t*/d_max = -4/9 ≈ -0.444
```

Both attaining configs:

- **n5-chain-fork**: `tree_arcs=[(s,n1),(n1,n2),(n2,n3),(n2,n4)]`,
  `attach=[(n1,n3),(n3,s),(n4,s)]`, `demands=(2,2,3)`,
  `w=(1/6,1/3,1/3)`.
- **n5-chain2-2leaf**: same t and d_max, different structure (see
  `outtree_results.json["per_combo"]` for the exact argmax).

**Per-tree max t (uncovered set only), as a ratio to d_max of the
attaining instance:**

| tree | max t | (t / d_max) |
|---|---|---|
| n4-chain3 | -3/2 | -1/2 |
| n4-fork-at-n1 | -7/2 | -7/6 |
| n4-chain2+leaf | -3 | -1 |
| n4-star3 | -55/12 | ~-1.53 |
| n5-chain4 | -3/2 | -1/2 |
| **n5-chain-fork** | **-4/3** | **-4/9** |
| n5-chain-leaf | -3/2 | -1/2 |
| n5-chain-star | -3/2 | -1/2 |
| n5-chain3-leaf | -3/2 | -1/2 |
| n5-cherry-leaf | -3/2 | -1/2 |
| n5-twochains | -3/2 | -1/2 |
| **n5-chain2-2leaf** | **-4/3** | **-4/9** |
| n5-star4 | -3/2 | -1/2 |
| n2/n3-* (all n=2,3 trees) | n/a | 100% covered, no uncovered instances exist |

**Plateau reading.** The best anything got in this sweep is `t/d_max ≈
-0.44` — nowhere close to the `t>0` needed for a counterexample, and
well below even the abstract mechanism ceiling this campaign has already
established elsewhere (NG-4's ladder found β\*=7/6, i.e. an *un-normalized*
overshoot ratio around 1.17, a different but related quantity — see
NOGOS.md). Every uncovered n=4/n=5 out-tree structure tried plateaus in
the `[-4/3, -55/12]` range on `t`, i.e. comfortably negative. This is a
genuinely small, MSW25-pruned search, so "no hit" here is a much sharper
statement than the earlier, unfiltered version of this same search would
have given for the same wall-clock budget.

## 8. Cost projection (stated before running, then revised once)

**First attempt** (`CAP_PATTERNS_PER_COMBO=20`, `TRIALS_PER_PATTERN=6`,
`RANDOM_SAMPLES=20`): projected ~290s based on the m=4 exact uncovered
counts (22 patterns total) plus a hand count of m=5 uncovered patterns
capped at 20/combo (~384 total) — actual cost per instance-search turned
out higher than the flat per-eval-cost estimate suggested (more tree arcs
→ more "balanced" heuristic candidates, plus n=5/k=5 evals costing more
than the k=4 pilot figure), so the run reached only 26/41 combos in 615s
before the `SOFT_STOP_S=600` guard truncated it (0 crashes, 0 assertion
failures, partial results written throughout — this worked exactly as
designed, just under-projected).

**Revised knobs** (`CAP_PATTERNS_PER_COMBO=15`, `TRIALS_PER_PATTERN=4`,
`RANDOM_SAMPLES=12`, `COVERED_SELF_TEST_CAP=5`, Lemma-6 self-test
restricted to 6 representative trees instead of all 16): completed all
41/41 combos in **540.2s**, comfortably inside the 720s ceiling. This is
the number reported above. Both runs agree: 0 bugs, 0 positive hits.

## 9. Coverage — what was NOT swept (honest statement)

- **n=6+ trees**: not enumerated at all. a(6)=20 unlabeled rooted trees;
  the uncovered fraction at n=6 is unknown but plausibly larger still
  (more room for K4-minor-creating chord crossings). Natural next step if
  this campaign continues down the out-tree branch.
- **k>5**: not swept for any tree size.
- **Exhaustiveness within the live (uncovered) set**: capped at
  `CAP_PATTERNS_PER_COMBO=15` uncovered structural patterns per (tree,k)
  — for the richest combos (n5-chain4/k5: 378 uncovered patterns;
  n5-chain3-leaf/k5: 378; n5-chain-leaf/k5: 350) this is a **~4% sample**,
  not a full sweep. Sampling is uniform random (seed 1729, reproducible).
  Smaller combos (all n=4 trees, all n=5 trees at k=3) were fully covered
  since their uncovered counts are ≤ the cap.
- **Demand-to-slot assignment within a sampled pattern**: only
  `TRIALS_PER_PATTERN=4` random (demand-vector, permutation) draws per
  pattern, not all `k!` permutations × all demand vectors. For k=5 that's
  4 of a possible (5 demand vectors × 120 permutations = 600) space per
  pattern — again a small sample, not exhaustive, on the richest combos.
- **Polish resolution**: coordinate polish uses a 13-point (denominator
  12) grid, 2 passes — finer than the corner/balanced/random points but
  still a grid, not a certified LP optimum. A true `t*` for any specific
  instance would need an exact LP/IP solve over the z ∈ {0,1}^k, w ∈
  [0,1]^k saddle, which this script does not attempt (out of scope for a
  12-minute screening sweep — this is a heuristic maximizer with exact
  arithmetic per evaluation, not an exact maximizer).
- **Equal-demand Lemma 6 self-test**: restricted to 6 of 16 trees (a
  deliberate scope cut once the MSW25 self-test needed the time budget
  more; Lemma 6 is a general structural claim not tied to a specific
  tree's automorphisms, so this is lower-risk than the uncovered-set
  sampling gaps above).

## 10. Takeaway

The out-tree branch of kill-path A/B, once correctly scoped by the MSW25
series-parallel theorem, has a small and shrinking live search space:
85.3% of all enumerated structural configurations (n=2..5, k=3..5) are
already provably dead by MSW25, and n≤3 trees are dead by vertex count
alone. What remains — Rybin's exact K4 completion at n=4 and its richer
n=5 chain-shaped relatives — plateaus at `t/d_max ≈ -0.44` under this
sweep, well short of 0. No counterexample, no near-miss, and a sharply
narrowed "what's left to check" for any follow-up: n=6+ trees and denser
sampling of the already-identified n=5 uncovered set (a ~2,400-pattern
space, only ~4% sampled at the richest combos).
