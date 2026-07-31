# Meet, join, and the closure operator: probing the Lattice Representation Hypothesis

**Started / finished:** 2026-07-31 · **Status:** done — **negative result**. The
opening thesis was refuted by this experiment's own adversarial and WordNet arms.
What survives is a set of caveats on the paper's experimental setting and one
measured effect (§5.2) whose proposed mechanism I could not verify.

**Origin.** A post by `@digthatdata.bsky.social` on the *Paper Skygest* Bluesky
feed, pointing at a paper "demonstrating how linear representations (i.e.
'embeddings' of the kind learned by DNNs) are connected to boolean logic and
formal concept analysis." That is
[**arXiv:2603.01227, "The Lattice Representation Hypothesis of Large Language
Models"** (Bo Xiong, Stanford; ICLR 2026)](https://arxiv.org/abs/2603.01227).

---

## TL;DR

I set out to show the paper's concept algebra has a **broken join**: FCA's meet
extent is a bare intersection (half-spaces are closed under intersection, so it
is exact), while its join extent is the *closure of a union* — and unions of
convex regions are not convex. Definition 7 of the paper writes the geometric
join as a literal set union, `A ∨ B := R(Y_A) ∪ R(Y_B)`, which is not a lattice
element; its own Appendix B Proposition 3 gives the correct closure form. I
predicted this asymmetry, measured it at scale, and pointed at the paper's
Figure 4, where join underperforms meet in **5 of 5** WordNet domains without
comment.

Then I ran an adversarial arm against my own claims. **Most of them died.**

| Claim | Verdict |
|---|---|
| Definition 7's join is not a lattice element | **Refuted.** Its "conic hull" clause makes it *exactly* `R(Y_A ∩ Y_B)` under linear independence, which holds throughout the paper's own experiments. Verified 96/96 by an independently written LP. |
| Overshoot is an error rate | **Refuted.** Those members *are* the join. In the WordNet task they are the gold label — the join of {dog, wolf} is `canine`, which contains foxes. |
| Meet is exact, join is lossy | **Reversed on the dominant metric, then confounded.** Under probe error the join degrades more slowly than the meet on Jaccard at every noise level — but join targets are ~12× larger, the direction flips on symmetric-difference error in one of two contexts, and the size-controlled version is reproduced by random-direction controls. |
| The 5/5 Figure 4 gap is explained by this | **Not established.** A difficulty control removes a third of it; the rest is not attributable to the join definition. |

**What survives** is smaller, and mostly consists of caveats on the setting:

1. **Both lattice operations are plain half-space intersections** — verified with
   0 identity violations. The join needs *fewer* constraints than the meet
   (0.83 vs 5.55 half-spaces). The only non-representable object in the
   neighbourhood is the plain union, which is not a lattice operation, and it is
   indeed recovered worst of the three. §5.1
2. **The join degrades more slowly under probe noise on Jaccard** — a real
   measurement, but with a large size confound and a mechanism I could not
   verify. §5.2–5.3
3. **WordNet's concept lattice is extremely thin** — 150 objects × 13 attributes
   give **15 concepts**, with 0/78 cross-cutting attribute pairs. This is a
   caveat on the paper's own experimental setting, not just mine. §5.4
4. **Learned attribute directions can be exactly antipodal** (coherence 1.0000),
   so "independent with probability 1 because `d` is large" is too quick — the
   paper's canonical-form assumption is never checked empirically. §5.4
5. **WordNet-shaped contexts are the worst case for the closure gap**, refuting
   the defence that hierarchical data is "nested and therefore fine." §4.3
6. **Equation 6's meet orientation looks inverted**, and unlike the Definition 7
   dispute this one is in the executed scoring path. It replicates strongly for
   the meet and *fails* to replicate for the join. §6

Plus three forensic observations about the paper that hold regardless of my
thesis, established by two independently-prompted readers: Figure 4's numbers
appear nowhere in the paper, 12 cells of Table 2 are arithmetically impossible,
and no released code computes Figure 4 at all.

---

## 1. The claim under test

The paper posits that attribute `m` holds of object `g` iff `⟨v_m, e_g⟩ ≥ τ_m`,
so every concept extent is the trace on `G` of an intersection of half-spaces —
a convex polyhedral cone after a canonical shift (its Definition 6). Concept
algebra is then geometric (its Definition 7):

> **Meet.** `A ∧ B := R(Y_A ∪ Y_B)` — "intersecting the half-spaces."
>
> **Join.** "the least upper bound in the lattice … `A ∨ B := R(Y_A) ∪ R(Y_B)`,
> which can be approximated by the conic hull spanned by the attribute
> directions of A and B."

Standard FCA (Ganter & Wille's Basic Theorem, and the paper's own Appendix B
Prop. 3) gives:

```
meet:  (A₁,B₁) ∧ (A₂,B₂) = ( A₁ ∩ A₂ ,      (B₁ ∪ B₂)'' )
join:  (A₁,B₁) ∨ (A₂,B₂) = ( (A₁ ∪ A₂)'' ,   B₁ ∩ B₂     )
```

The extents form a **closure system**: closed under intersection, *not* under
union. So the meet's extent needs no closure and the join's does. That
asymmetry is real and is the thing I set out to exploit.

## 2. Method

`src/fca.py` is the shared core — derivation operators, Ganter's NextClosure
concept enumeration, the geometric realization `realize(E, V, τ)`, and two
**two-sided calibration gates** (`METHODS.md` principle 2): `assert_meet_closed`
(known-good — must always pass, it is a theorem) and `assert_join_overshoots`
(known-bad — must find the phenomenon, or the context is degenerate for this
study). Verified against brute-force enumeration over the attribute powerset:
59 concepts both ways.

Six arms, each with its own JSON in `results/`:

| Arm | What | File |
|---|---|---|
| A | Synthetic half-space realization; sweeps over dimension, direction coherence, density | `arm_a_synthetic.py` |
| B | WordNet sub-hierarchies + real MiniLM embeddings + held-out linear probes | `arm_b_wordnet.py` |
| C | Five LLM-elicited formal contexts (~55 objects × 13 attrs each) | `arm_c_llm_contexts.py` |
| D | Are the three join operators the same object? | `arm_d_join_operators.py` |
| E | Does the gap survive on realistic context shapes and geometry? | `arm_e_context_shape.py` |
| F | **Checking the adversarial review's own claims** | `arm_f_adversarial_response.py` |

`THEORY.md` (1068 lines, separate agent) carries the proofs: the minimal
overshoot counterexample at `(|G|,|M|) = (3,2)` realized in `d=1`, the exact
emptiness characterization, the trilemma with all three pairs shown satisfiable,
and the Cover/Warren counting bounds on how many contexts `d` dimensions can
realize at all.

## 3. What the synthetic arms found

Across **9,615,370 concept pairs** in every configuration of Arm A, total meet
phantoms = **0**. Exact, always. This is Monte Carlo verification of a theorem,
not a discovery — see §5.

Join overshoot at the baseline (`d=64`, 12 attributes, `p=0.5`, 200 objects, 12
seeds): mean **0.603 ± 0.030**, median 0.656, p90 0.842. Two concepts covering
21.8 objects between them join to an extent of 67.8 — a **3.1× inflation**.

Three sweeps (`figures/sweeps.png`):

- **Dimension.** As `d` falls 64 → 2, overshoot/|join| falls 0.60 → 0.20 — but
  that is a **normalization artifact**. Per object *at risk* the rate rises
  monotonically 0.28 → 0.61 (Spearman ρ = −1.000). The lattice is dissolving:
  concepts 2050 → 83, realizable intents 50% → 2%. Reporting only the first
  normalization would have concluded "low dimension is benign," which is wrong.
  Both curves are plotted.
- **Coherence** at fixed `d=64`, full rank throughout: overshoot falls 0.60 →
  0.10 as coherence rises 0.33 → 0.93. Positively correlated attributes
  co-occur, so more half-spaces survive to constrain the closure.
- **Density.** Monotone; sparse is worst (0.78 at `p=0.1`).

Rank deficiency produces **no discontinuity** — the curves pass straight through
`d = |M|` where linear independence fails. The paper's canonical-form
precondition degrades continuously rather than breaking.

## 4. The adversarial arm, and what it killed

I asked a separately-prompted agent to attack the critique rather than the
paper. It landed. I then re-checked its load-bearing claims with my own code
(Arm F), because a reviewer can be wrong too.

### 4.1 The cone identity — my central claim is dead

`R(Y) = {v : ⟨v,d_m⟩ ≥ 0 ∀m ∈ Y}` is the dual cone `C_Y*`. Then
`cone-hull(R(Y_A) ∪ R(Y_B)) = R(Y_A) + R(Y_B) = (C_A ∩ C_B)*`, and
`R(Y_A ∩ Y_B) = (C_{A∩B})*`. These coincide iff `C_A ∩ C_B = C_{A∩B}`, which
holds whenever `{d_m : m ∈ Y_A ∪ Y_B}` is **linearly independent**.

My independent LP check (Minkowski-sum membership, written from the dual-cone
definition, sharing no code with the region construction):

| case | points in conic hull | verdict |
|---|---|---|
| independent, `d=8`, 5 dirs | 96/96 | **EQUAL** |
| independent, `d=20`, 10 dirs | 96/96 | **EQUAL** |
| independent, `d=64`, 14 dirs | 96/96 | **EQUAL** |
| dependent, `d=2`, 4 dirs | 43/96 | strict |
| dependent, `d=3`, `d₃ = d₁+d₂` | 34/96 | strict |

The paper runs at `|M| ≤ 184` and `d ≥ 3072`. Its attribute directions are
linearly independent with probability 1. **So Definition 7, read with its own
conic-hull clause, denotes exactly the object I nominated as "the correct
join."** The bare `∪` is loose notation repaired in the same sentence. The
"contradicts its own Appendix B" charge does not stand.

What remains is a real but modest editorial point, worth sending to the author:
the paper calls the conic hull an *approximation*, and under its own stated
independence assumption it is an **equality**. That is a strengthening the paper
could have claimed and didn't.

### 4.2 "Phantoms" are not errors

The overshoot metric measures the size of the closure gap. A textbook-correct
FCA join scores 0.60 on it, because the join extent *is* the closure. Calling
those members phantoms describes the least upper bound, correctly computed, as
a failure. In the paper's actual task it is worse than that: gold joins are
least common hypernyms, so the "phantoms" are the gold label.

**The 0.60 and the 82–92% must be read as descriptive statistics about how far
a lattice join sits from plain disjunction — not as an error rate, and not as
evidence against the paper.** They are relabelled that way throughout, including
in the figure titles.

### 4.3 Where the review was wrong

The review defended the paper by arguing WordNet sub-hierarchies are "strongly
nested, near a chain," where the closure gap vanishes. A chain does vanish. A
**tree is not a chain** — two leaves in different branches have incomparable
extents, and their union closes up to their common ancestor. Arm F2, five seeds:

| context shape | union not closed | overshoot |
|---|---|---|
| chain (totally nested) | 0.000 | 0.000 |
| **tree, depth 4 branch 3** | **0.928** | **0.713** |
| **tree, depth 3 branch 4** | **0.907** | **0.735** |
| iid Bernoulli `p=0.3` | 0.884 | 0.651 |
| iid Bernoulli `p=0.5` | 0.913 | 0.495 |

Hypernym trees — the paper's own data shape — are the **worst** case measured,
worse than the iid controls the review used to argue vacuity. The statistic is
not an artifact of unrealistic random contexts. (This does not rescue the
*interpretation* in §4.2; it only establishes that the quantity is relevant to
the paper's setting.)

### 4.4 Figure 4, difficulty-normalized

The Random baseline contains no method, so its join/meet ratio isolates task
difficulty. It is a remarkably uniform **0.976–0.979** across all five domains —
the join task is intrinsically ~2.2% harder in MRR. Subtracting that:

| domain | random | mean | ours | excess (ours − random) |
|---|---|---|---|---|
| WN-Animal | 0.978 | 0.928 | 0.934 | −0.044 |
| WN-Plant | 0.976 | 0.932 | 0.941 | −0.036 |
| WN-Food | 0.979 | 0.934 | 0.940 | −0.039 |
| WN-Event | 0.978 | 1.000 | 0.907 | −0.071 |
| WN-Cognition | 0.978 | 1.095 | 0.955 | −0.022 |

An excess join deficit survives difficulty normalization in **5/5 domains**
(the review claimed it collapses to 2/5; using the Random control rather than
Mean, it does not). **But** the Mean baseline — which has no regions, no
half-spaces, no concept algebra — shows a comparable excess in the three
physical domains. So the effect is not specific to the geometric join operator,
and **cannot be attributed to it**. Only in the two abstract domains is the
deficit unique to the paper's method, and the paper independently documents an
unrelated weakness there.

Honest verdict: the 5/5 join deficit is real, unremarked by the paper, and
**unexplained**. My explanation is not established, and is not the leading
candidate.

## 5. The join degrades more slowly under probe noise — with a size confound

This is the finding I did not expect. It is real but narrower than my first
reading of it, and §5.3 states what it does not support.

Arm B builds two WordNet contexts (`tree`: 150 objects × 13 ancestor attributes
from `animal/vehicle/plant` roots, density 0.221, zero cross-cutting attribute
pairs; `cross`: 111 × 13, density 0.248, 13/78 cross-cutting), embeds objects
with MiniLM, and fits **held-out** linear probes. Probe quality, 8 seeds, three
text variants, two controls:

| variant | logreg AUC (tree) | logreg AUC (cross) | random-direction control | shuffled-label control |
|---|---|---|---|---|
| lemma only | 0.869 | 0.688 | 0.530 / 0.498 | 0.479 / 0.473 |
| lemma + gloss | 0.974 | 0.943 | 0.525 / 0.507 | 0.481 / 0.483 |
| gloss only | 0.977 | 0.949 | 0.523 / 0.503 | 0.491 / 0.490 |

So the LRH threshold model does fit — strongly with glosses, moderately from
lemma alone — and both controls sit at chance. (Reported for all three variants
rather than the best one; the gloss variants carry most of the signal, which is
itself worth noting, since glosses contain the hypernym in prose.)

Then the key sweep: inject probe error at a known per-cell rate and measure
recovery of the true meet and true join (`figures/noise_reversal.png`).

| flip rate | meet J (tree) | join J (tree) | meet J (cross) | join J (cross) |
|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.01 | 0.954 | **0.987** | 0.967 | **0.977** |
| 0.02 | 0.935 | **0.970** | 0.854 | **0.922** |
| 0.05 | 0.835 | **0.939** | 0.753 | **0.875** |
| 0.10 | 0.745 | **0.867** | 0.657 | **0.782** |
| 0.20 | 0.597 | **0.750** | 0.527 | **0.675** |

**On Jaccard, the join degrades more slowly than the meet at every nonzero noise
level, in both contexts.** The plain union — the object my critique treated as
the "logically correct" target — tracks the meet, not the join, and is recovered
worst of the three (`tree` error fraction 0.065, vs meet 0.021 and join 0.018).
That much is exactly what the theory predicts: the union is the only one of the
three that is not an intersection of half-spaces.

### 5.1 Why the premise was wrong in the first place

Arm B checked the identity rather than assuming it. For genuine concepts
`attrs_of(A₁ ∪ A₂) = B₁ ∩ B₂`, so

```
meet extent = A₁ ∩ A₂     = objs_of(B₁ ∪ B₂)
join extent = (A₁ ∪ A₂)'' = objs_of(B₁ ∩ B₂)
```

**Both are plain intersections of half-spaces** — `identity_violations_meet = 0`
and `identity_violations_join = 0` across all 30 configurations × 8 seeds. The
closure is not something half-spaces must over-approximate; the closure is
precisely what lands the join back on a representable set. The only structural
difference is constraint *count*: the meet intersects 5.55 / 5.84 half-spaces on
average, the join only 0.83 / 1.09, and 43% / 42% of join targets need zero
constraints (the top of the lattice, all of `G`).

### 5.2 The size confound

Join extents are **~12× larger** than meet extents (43.1 vs 3.5 objects in
`tree`; 30.6 vs 2.4 in `cross`), and 66% / 58% of true meets are *empty*.
Jaccard is generous to large targets. Switch to symmetric-difference error and
the answer flips in one of the two contexts:

| context | meet J | join J | meet err | join err |
|---|---|---|---|---|
| `tree` | 0.839 | **0.932** | 0.0206 | **0.0180** |
| `cross` | 0.711 | **0.784** | **0.0276** | 0.1102 |

In `cross` the join is **4× worse** on error fraction while simultaneously
better on Jaccard. Both metrics are defensible; they disagree because of target
size, not because of geometry.

### 5.3 What kills the size-controlled version

Comparing recovery of `objs_of(B)` binned by `|B|` puts meet and join on one
axis. At matched half-space count, meet − join Jaccard is positive in **all 30
configurations** (+0.069 `tree`, +0.126 `cross`). That looks like the asymmetry
I predicted. Two controls remove it:

1. **Random directions produce the *largest* matched-`k` gap** — larger than any
   real probe. A gap that widens for a chance-level probe is not evidence about
   embedding geometry.
2. **Under i.i.d. incidence noise with the lattice held fixed, the matched-`k`
   gap has no stable sign** (−0.054 to +0.058, mean ≈ 0).

And the mechanism I proposed is contradicted outright: the per-pair correlation
between a pair's intrinsic overshoot and its join recovery error is
**−0.44 in `tree`** (+0.39 in `cross`, so not even consistent in sign). In the
tree context the pairs whose join travels *furthest* from the union are the
*easiest* to recover, because their join is simply all of `G`.

**So: the join's slower degradation under noise is a real measurement, and the
"closure absorbs probe error" story is a plausible but unverified explanation
for it. It is not established, the size confound is large, and the
size-controlled version does not survive its own controls.**

### 5.4 Two caveats on the setting itself — including one for the paper

- **WordNet noun hypernymy is a bare tree.** The naive context has **0 of 78
  cross-cutting attribute pairs**, and 150 objects × 13 attributes yields only
  **15 concepts** out of 8192 possible intents. Even a selection deliberately
  maximising overlap reaches 13/78 and 22 concepts. Any WordNet-based evaluation
  of this hypothesis — **including the paper's own** — is testing an extremely
  thin lattice, where most meets are empty and ~42% of joins are the top element.
- **Learned attribute directions can be exactly antipodal.** Mutual coherence of
  the fitted directions is **1.0000** in `tree` (± 6e-16), for every probe method
  and text variant, because `living_thing.n.01` and `artifact.n.01` are exactly
  complementary on that object set — the probes are `y` and `¬y`. `cross` reaches
  0.920, against 0.139 for random directions.

  This qualifies §4.1. The defence that the paper's directions are "linearly
  independent with probability 1" because `|M| ≤ 184 ≪ d` assumes *generic*
  directions. Directions **learned** from a taxonomy that partitions its universe
  are not generic and can be exactly dependent regardless of how large `d` is.
  The conic-hull identity needs independence, so a partitioning taxonomy is
  precisely where it can fail. This does not revive the original claim — exact
  antipodality is a special case, and the paper's attributes are mostly not
  complementary — but it does mean the dimension-counting defence is too quick,
  and the paper never checks the assumption empirically.

## 6. Equation 6's orientation

The paper's *operational* definition — the one behind every number in Figure 4 —
is coordinatewise: `π_{A∧B}(m) = min{π_A, π_B}`, `π_{A∨B}(m) = max{π_A, π_B}`.
Under its own model the meet's intent is `Y_A ∪ Y_B` (more attributes) and the
join's is `Y_A ∩ Y_B` (fewer), which suggests the assignment is inverted.

Arm F4, 400 random lattices, crisp attribute profiles, cosine to the true
extent's profile:

| | min | max |
|---|---|---|
| true **meet** profile | 0.591 | **0.626** |
| true **join** profile | **0.920** | 0.924 |

- **meet:** `max` beats `min` in **96.5%** of trials — the inversion replicates
  strongly.
- **join:** `min` beats `max` in only **59.0%** of trials, and the mean cosines
  are effectively tied. **This half does not replicate.**

The adversarial agent reported 99.5% for both halves. My independent
implementation reproduces the meet result and **not** the join result; the
likely cause is a different definition of the attribute profile (mine is the
mean attribute indicator over the extent; theirs was unspecified). Reported as a
discrepancy rather than resolved — this needs the authors' code, which for this
figure does not exist (§7).

## 7. Forensics on the paper itself

Two agents read the paper independently, the second explicitly forbidden from
seeing the first's analysis. Both, working from the LaTeXML-emitted SVG bar
geometry, recovered Figure 4's values **identically to three decimals**:

| Ours | Animal | Plant | Food | Event | Cognition |
|---|---|---|---|---|---|
| meet | 0.547 | 0.558 | 0.565 | 0.505 | 0.492 |
| join | 0.511 | 0.525 | 0.531 | 0.458 | 0.470 |

Caveat kept explicit: these are **decoded from the figure's vector geometry, not
printed in the paper**. All 30 bars round-trip to clean 3-decimal values, which
is good evidence the calibration is right, but the two decodings share a method,
so this is replication and not an independent check of the approach.

Three observations that stand independently of my thesis:

1. **The Figure 4 numbers appear nowhere in the paper.** Four tables, none of
   them meet/join MRR. The bar chart is the sole record.
2. **12 of 45 cells in Table 2 are arithmetically impossible** — F1 strictly
   below both its own precision and recall, which a harmonic mean cannot be.
   WN-Food is impossible in 6 of its 9 rows; the worst cell is Gemma/Random/Food
   at 39.1 where P=53.4, R=50.8 forces 52.07. The released code uses sklearn's
   `f1_score`, which cannot produce these, so they did not come from that path.
3. **No released code computes Figure 4.** `join_meet.py` (57 lines) builds gold
   labels from NLTK WordNet for five hardcoded pairs and writes two CSVs — no
   model, no embeddings, no ranking, no MRR. Equations 6–9 are unimplemented
   anywhere in the repository. The concept-algebra experiment, one of the
   paper's three headline contributions, has no released implementation.

## 8. Arm C is a null result, and one of the failures is mine

Five LLM-elicited formal contexts were built (animals 55×14, foods 56×13,
instruments 55×14, occupations 59×14, vehicles 54×14; all with density gates,
duplicate-row checks, and documented judgement calls in `data/contexts/`). They
are good data and are reused by Arm D.

The Arm C *analysis* failed in two ways, both reported rather than dropped:

1. **My meet metric was vacuous by construction.** `meet_extent(ctx, A₁, A₂)`
   returns `A₁ & A₂` and ignores the context, so comparing it between the true
   and realized contexts trivially returns Jaccard 1.000 for every domain. It
   tests nothing. The bug is left in the record; Arm B's noise sweep is the
   measurement that actually works.
2. **The learned probes are indistinguishable from random directions.** Join
   Jaccard, probe vs. random-direction baseline: animals 0.597/0.596, foods
   0.507/0.509, instruments 0.591/0.586, occupations 0.519/0.522, vehicles
   0.571/0.560. Held-out probe accuracy was only 0.68–0.75 on bare object names.
   With no separation from the control, this arm cannot support any conclusion
   and is not used for one.

## 9. What I would tell the author

Framed as a strengthening rather than a takedown:

- Definition 7's `∪` should be the conic hull, and "can be approximated by"
  understates it: under the linear independence Proposition 1 already assumes,
  `cone-hull(R(Y_A) ∪ R(Y_B)) = R(Y_A ∩ Y_B)` **exactly**, with an explicit
  counterexample once the directions are conically dependent. Small, clean,
  worth stating.
- Equation 6's meet orientation looks inverted relative to Proposition 3.
  Raised as a question with numbers attached, since the join half of the test
  did not replicate.
- Figure 4's join deficit is 5/5 and survives a Random-baseline difficulty
  control. It deserves a sentence, whatever its cause.
- The canonical-form assumption (linearly independent attribute directions) is
  never checked empirically. Directions *learned* from a partitioning taxonomy
  need not be generic — I measured mutual coherence of exactly 1.0000 on a
  WordNet context where two attributes were complementary. `d >> |M|` does not
  by itself secure the assumption.
- WordNet noun hypernymy yields a very thin lattice (15 concepts from
  150 objects x 13 attributes; 0/78 cross-cutting attribute pairs). Reporting
  concept-algebra results on it without that caveat overstates what the
  evaluation can distinguish.
- Table 2's 12 impossible F1 cells need correcting.
- Figure 4 needs both a number table and a released implementation.

## 10. Reusable findings

- **Both FCA lattice operations are half-space intersections; the plain union
  is the odd one out.** `meet = objs_of(B1 u B2)`, `join = objs_of(B1 n B2)` —
  the closure is what makes the join representable, not what breaks it. If you
  are reasoning over learned attribute probes, the operation to distrust is set
  union, which is not a lattice operation at all. §5.1
- **Compare set-recovery metrics only at matched target size.** Jaccard and
  symmetric-difference error gave opposite meet-vs-join answers on the same
  data, purely because join extents were 12x larger. Report both, or bin by
  constraint count — and then check the gap against a random-direction control,
  which in this case reproduced it entirely. §5.2-5.3
- **Never report a normalized rate without its denominator's behaviour.** Arm
  A's dimension sweep reverses sign between two defensible normalizations, and
  the one that looks natural gives the wrong conclusion. §3.
- **A closure gap is a property of the context, not an error.** Before treating
  `|S'' \ S|` as a defect, check whether the closure *is* the intended answer.
  §4.2.
- **Trees are not chains.** "Hierarchical data is nested, so closure gaps
  vanish" is false; hypernym trees are the worst case. §4.3.
- **Adversarially reviewing your own thesis before writing it up is worth the
  compute.** This experiment's headline claim would have been wrong, and the
  interesting result — §5 — only became visible once the wrong one was cleared.

## Files

```
THEORY.md                          proofs; minimal counterexample; trilemma; counting bounds
src/fca.py                         shared FCA core + geometric realization + calibration gates
src/arm_a_synthetic.py             dimension / coherence / density sweeps
src/arm_b_wordnet.py               WordNet + MiniLM + held-out probes + noise sweep
src/arm_c_llm_contexts.py          LLM-elicited contexts (null result, see §8)
src/arm_d_join_operators.py        the three join operators compared
src/arm_e_context_shape.py         tree vs chain vs iid; anisotropic geometry
src/arm_f_adversarial_response.py  checking the adversarial review's own claims
src/make_figures.py
data/contexts/*.json               five hand-built formal contexts
results/*.json                     every number reported above, with its config
figures/noise_reversal.png         §5, the result that reversed the thesis
figures/sweeps.png                 §3
```

Reproduce: `cd src && python3 fca.py && python3 arm_a_synthetic.py && …`
Needs `nltk` (+`wordnet`), `sentence-transformers`, `scikit-learn`, `scipy`,
`matplotlib`. No API keys; MiniLM runs on CPU.
