# Meet, Join, and the Dimension Budget

**Theory arm of the Lattice Representation Hypothesis experiment.**

Target: Bo Xiong, *The Lattice Representation Hypothesis of Large Language
Models*, arXiv:2603.01227. The paper proposes that LLM embedding geometry
encodes a Formal Concept Analysis concept lattice via linear attribute
directions with separating thresholds, that meet and join are both realized
geometrically, and that a canonical form exists when the attribute directions
are linearly independent.

This document argues that the second claim is false as stated and that the
third claim, where it holds, drains the hypothesis of empirical content.

Notation follows `src/fca.py` throughout: a context is `Context(I)`,
derivations are `A'` (`attrs_of`) and `B'` (`objs_of`), closures are `A''`
(`close_objs`) and `B''` (`close_attrs`), and `realize(E, V, tau)` builds the
context induced by a geometry.

---

## 0. Status legend

Every non-trivial assertion below carries one of these tags.

| Tag | Meaning |
|---|---|
| **[P]** | Proved here, complete argument given. |
| **[P/sk]** | Proof sketch given; the gap is routine but I did not write it out. |
| **[V]** | Verified numerically only (throwaway scripts under `/tmp/lrh-theory/`, listed in §7). Not proved. |
| **[P+V]** | Proved *and* independently checked numerically. |
| **[C]** | Conjecture. Consistent with pilots, no proof. |
| **[X]** | I could not establish this. Stated so the empirical arms do not assume it. |

Section 7 lists every script, what it checked, and the raw numbers. Section 8
lists everything I failed to prove.

---

## 1. Setup: the geometric realization

### 1.1 The model

**Definition 1.1 (realization).** A *realization* in dimension `d` is a triple
`(E, V, tau)` with `E ∈ R^{n×d}` (object embeddings `e_g`), `V ∈ R^{m×d}`
(attribute directions `v_m`), `tau ∈ R^m` (thresholds). It induces the formal
context `K = (G, M, I)` with `G = {1..n}`, `M = {1..m}` and

```
    g I m   <=>   <v_m, e_g>  >  tau_m .
```

This is exactly `realize(E, V, tau)` in `src/fca.py`. Note the **strict**
inequality — §1.4 shows it does not matter.

For `m ∈ M` write the **open half-space**

```
    H_m  =  { x ∈ R^d : <v_m, x> > tau_m } ,
```

and for `B ⊆ M` the **open polyhedron**

```
    P_B  =  ∩_{m ∈ B} H_m ,        P_∅ = R^d .
```

`P_B` is an intersection of finitely many open half-spaces: open, convex,
possibly empty. Call `L(M) = { P_B : B ⊆ M }` the **language** of the
realization. `|L(M)| ≤ 2^m` and `L(M)` is closed under intersection.

Write `G ∩ P` for the **trace** of a region on the object set, meaning the
index set `{ g ∈ G : e_g ∈ P }`. Traces, not regions, are all we ever observe.

**Caveat 1.2 (objects are indices, not points).** The map `g ↦ e_g` need not be
injective. If `e_g = e_h` then rows `g` and `h` of `I` are identical and no
extent can separate them. All statements below are about the index set `G`;
"`g ∈ conv(A)`" abbreviates "`e_g ∈ conv{ e_h : h ∈ A }`".

### 1.2 Extents are traces of the language

**Theorem 1.3.** In a realized context, for every `B ⊆ M`,

```
    B'  =  G ∩ P_B ,
```

and consequently every concept extent `A` satisfies `A = A'' = G ∩ P_{A'}`,
i.e. every extent is the trace of the intersection of exactly the half-spaces
of its own intent. **[P]**

*Proof.* `objs_of(B) = { g : g I m for all m ∈ B } = { g : <v_m, e_g> > tau_m
for all m ∈ B } = { g : e_g ∈ ∩_{m∈B} H_m } = G ∩ P_B`. If `(A, B)` is a
concept then `A = B'` and `B = A'`, so `A = G ∩ P_{A'}`. ∎

**Corollary 1.4 (closure = "intersect every half-space you are inside").** For
any `A ⊆ G`, `A'' = G ∩ P_{A'}` where `A'` is the set of attributes whose
half-space contains all of `A`. So `A''` is the trace of the *smallest member
of `L(M)` containing `A`*. **[P]**

*Proof.* `A'' = objs_of(attrs_of(A)) = G ∩ P_{A'}` by Theorem 1.3. `P_{A'}` is
the intersection of all `H_m ⊇ A`, hence the smallest element of `L(M)`
containing `A` (elements of `L(M)` containing `A` are exactly the `P_B` with
`B ⊆ A'`, and these all contain `P_{A'}`). ∎

Corollary 1.4 is the whole geometric content of the closure operator. It also
already shows where the trouble will be: the *smallest* member of `L(M)`
containing a set is what the geometry gives you, whether or not that is what
you wanted.

### 1.3 Convexity forces phantoms

**Lemma 1.5 (hull lemma).** For any `A ⊆ G` and any `g ∈ G`, if
`e_g ∈ conv{ e_h : h ∈ A }` then `g ∈ A''`. **[P+V]**

*Proof.* `P_{A'}` is convex (intersection of half-spaces) and contains `e_h`
for every `h ∈ A` (by definition of `A'`). Hence it contains the convex hull,
so `e_g ∈ P_{A'}`, so `g ∈ G ∩ P_{A'} = A''` by Corollary 1.4. ∎

Numerically: 4465 (object, extent-pair) cases in random `d = 2` realizations,
zero violations (`check2_geometry.py`).

**Corollary 1.6.** `G ∩ conv(A) ⊆ A''` for every `A ⊆ G`. In particular, no
realization in any dimension, with any attribute directions, can have an extent
that omits an object lying in the convex hull of the extent. **[P]**

This is a hard, geometry-only constraint that survives every hypothesis the
paper makes and every hypothesis it does not. It is the source of the
lower bound on overshoot in §3.

**Warning 1.7.** The converse of Lemma 1.5 is **false**: `A''` can contain
objects far outside `conv(A)`. In the same random sample, 638 of 723 phantoms
were *outside* the hull versus 85 inside. The hull is a lower bound on
overshoot, never an estimate of it. **[V]**

### 1.4 Open versus closed half-spaces

**Proposition 1.8.** On a finite object set the open/closed distinction is
immaterial. Given a realization with open half-spaces there exist thresholds
`tau*` with `tau*_m ≥ tau_m` such that

```
    { g : <v_m, e_g> > tau_m }  =  { g : <v_m, e_g> ≥ tau*_m }   for every m,
```

so the same context is realized by closed half-spaces, and conversely. **[P]**

*Proof.* Fix `m`. Let `S = { <v_m, e_g> : g ∈ G, <v_m, e_g> > tau_m }`. If
`S = ∅` take `tau*_m` larger than `max_g <v_m, e_g>`. Otherwise `S` is finite
and nonempty; let `s = min S > tau_m` and set `tau*_m = s`. Then
`<v_m,e_g> ≥ tau*_m` iff `<v_m,e_g> ≥ s` iff `<v_m,e_g> > tau_m`. The converse
direction is the same argument with `tau*_m = (s + t)/2` for `t` the largest
value strictly below `s`, or any smaller value if none exists. ∎

**Warning 1.9.** Proposition 1.8 is about *traces*. The regions themselves are
genuinely different: `∩ H_m` is open, `∩ closure(H_m)` is closed, and they have
different boundaries, different closures, and different Helly behaviour on
measure-zero sets. Any argument that passes through the ambient topology must
pick one and stay there. Below, `P_B` is always the **open** version.

### 1.5 Four claims that are true of the polyhedra but not of the traces (or vice versa)

These are the places I expect a careless treatment to slip. Each is stated with
its truth value.

**(T1) "Every trace `G ∩ P` of a convex polyhedron `P` is a concept extent."
FALSE. [P]** Extents are traces of members of `L(M)` only. Take `d = 1`,
`G = {0, 1, 2}`, one attribute `m` with `H_m = (0.5, ∞)`, so extents are
`{1,2}`, `G`. The polyhedron `(1.5, ∞)` has trace `{2}`, which is not an
extent. Arbitrary half-spaces cut out arbitrary subsets; the *given* ones do
not.

**(T2) "The polyhedron of an intent is determined by the extent." FALSE. [P]**
Many members of `L(M)` share a trace. This matters because "the concept is a
polyhedron" is only true up to trace-equivalence, and trace-equivalence classes
can be enormous.

**(T3) "`P_{(B1 ∪ B2)''} = P_{B1 ∪ B2}`." FALSE as ambient sets, TRUE as
traces. [P+V]** `B1 ∪ B2 ⊆ (B1 ∪ B2)''` so the left side is contained in the
right, and `objs_of` of both is the same set by definition of `''`. But
containment can be strict in `R^d`. Witness (verified in `check6_budget.py`):
`d = 2`, attributes `a: x > 0`, `b: y > 0`, `c: x + 2y > 3`;
`G = { (2,2), (-1,-1) }`. Then `{a,b}'' = {a,b,c}`, the traces of `P_{a,b}` and
`P_{a,b,c}` are both `{(2,2)}`, but `(0.1, 0.1) ∈ P_{a,b} \ P_{a,b,c}`.

**(T4) "A concept lattice determines its realization's geometry." FALSE. [P]**
Immediate from (T2) and (T3) and from Theorem 5.2 below (when `rank(V) = m`,
*every* boolean matrix is realizable with those directions).

**(T5) "Helly bounds the size of minimal unsatisfiable attribute sets." TRUE
ambient, UNOBSERVABLE on `G`. [P]** By Helly's theorem, for a finite family of
convex sets in `R^d`, if every `d+1` of them have a common point then all do.
Contrapositive: if `P_B = ∅` then some `B0 ⊆ B` with `|B0| ≤ d + 1` already has
`P_{B0} = ∅`. So a `d`-dimensional realization admits no minimal *geometrically*
inconsistent attribute set of size `> d + 1`. **But `B' = ∅` does not imply
`P_B = ∅`** — the polyhedron can be a perfectly good nonempty region that
simply contains no sampled object. Helly therefore constrains something the
data never shows us. See §8 for why I could not turn this into a test.

---

## 2. The meet is free

**Theorem 2.1 (meet closure).** If `A1` and `A2` are concept extents of a
context `K` then `A1 ∩ A2` is a concept extent. No geometric hypothesis is
used — this holds for every formal context. **[P]**

*Proof.* Write `A1 = B1'`, `A2 = B2'` with `B1 = A1'`, `B2 = A2'`. Directly
from the definition of `objs_of`,

```
    B1' ∩ B2' = { g : g I m ∀ m ∈ B1 } ∩ { g : g I m ∀ m ∈ B2 }
              = { g : g I m ∀ m ∈ B1 ∪ B2 }
              = (B1 ∪ B2)' .
```

Any set of the form `X'` is an extent, because the derivation operators form a
Galois connection and hence `X''' = X'` for all `X`. Setting `X = B1 ∪ B2`
gives `(A1 ∩ A2)'' = A1 ∩ A2`. ∎

This is `assert_meet_closed` in `src/fca.py`, and it is a theorem, which is why
that function is written as a calibration gate rather than a measurement.

**Theorem 2.2 (the meet is realized exactly).** In a realized context, with
`A_i = G ∩ P_{B_i}`,

```
    A1 ∩ A2  =  G ∩ (P_{B1} ∩ P_{B2})  =  G ∩ P_{B1 ∪ B2} ,
```

so the meet's extent is the trace of an explicit member of `L(M)`, obtained by
intersecting the half-spaces of the two intents. **[P]**

*Proof.* Two facts. (i) The trace operator commutes with intersection:
`G ∩ (P ∩ Q) = (G ∩ P) ∩ (G ∩ Q)` for any regions, since membership of `e_g` in
an intersection is the conjunction of memberships. (ii) `L(M)` is closed under
intersection with `P_{B1} ∩ P_{B2} = P_{B1 ∪ B2}`, immediately from the
definition of `P_B` as an indexed intersection. Combine. ∎

**Remark 2.3.** The content of Theorem 2.2 is one line: *finite intersections of
half-spaces are finite intersections of half-spaces.* Nothing else is used. In
particular the meet needs

- no linear independence of `{v_m}`,
- no bound relating `d` and `|M|`,
- no general-position assumption on `E`,
- no assumption that `P_{B1} ∩ P_{B2} ≠ ∅`.

The meet is free because the language was chosen to be an intersection-closed
family, and the concept lattice's meet is intersection. The paper is right about
the meet, and being right about it costs nothing.

**Remark 2.4 (an asymmetry hiding in plain sight).** The meet's *intent* is
`(B1 ∪ B2)''`, which can be strictly larger than `B1 ∪ B2` — closure appears on
the intent side of the meet exactly as it appears on the extent side of the
join. The difference is that the *extent* is what the geometry has to produce
as a region, and on the meet side the extent needs no closure. The lattice is
symmetric; the realization is not, because the realization only ever renders
extents as regions. `src/fca.py`'s docstring says this correctly.

---

## 3. The join is not

### 3.1 The language is not closed under union

**Proposition 3.1.** `L(M)` is not closed under union: there exist `B1, B2 ⊆ M`
such that `P_{B1} ∪ P_{B2}` is not convex, hence is not `P_B` for any `B ⊆ M`
and not the intersection of *any* family of half-spaces. **[P]**

*Proof.* Take `d = 1`, `H_1 = (-∞, 0)` (i.e. `v_1 = -1, tau_1 = 0`),
`H_2 = (1, ∞)`. Then `P_{{1}} ∪ P_{{2}}` contains `-1` and `2` but not `0.5`,
so it is not convex. Every element of `L(M)` is an intersection of convex sets
and hence convex. ∎

**Warning 3.2 (this does not immediately settle the question we care about).**
Proposition 3.1 is about ambient regions. What the hypothesis needs is only that
the *trace* `A1 ∪ A2` be the trace of some member of `L(M)`. A finite `G` can
easily fail to sample the non-convexity: in Proposition 3.1's geometry, if no
object embeds into `[0, 1]` then `A1 ∪ A2 = G` and `G = G ∩ P_∅` is a perfectly
good extent. **Non-closure of the region class does not by itself prove
non-closure of the trace class.** Any argument that stops at Proposition 3.1 is
incomplete. The correct statement is Theorem 3.5.

### 3.2 The lattice join, and the overshoot

The lattice join of concepts `(A1,B1)` and `(A2,B2)` is
`((A1 ∪ A2)'', B1 ∩ B2)` (Ganter & Wille, Basic Theorem on Concept Lattices).
This is `join_extent` in `src/fca.py`.

**Proposition 3.3 (the join *is* representable).** `(A1 ∪ A2)'' = (B1 ∩ B2)' =
G ∩ P_{B1 ∩ B2}`: the join's extent is the trace of the intersection of exactly
the *shared* half-spaces of the two intents. **[P]**

*Proof.* `(A1 ∪ A2)' = A1' ∩ A2' = B1 ∩ B2` (an attribute is shared by all of
`A1 ∪ A2` iff it is shared by all of `A1` and by all of `A2`). Apply `'` and
Theorem 1.3. ∎

So the join is in the language. It is just not the union.

**Definition 3.4 (overshoot).** For concepts `(A1,B1), (A2,B2)`,

```
    Over(A1, A2)  =  (A1 ∪ A2)'' \ (A1 ∪ A2)  =  (B1 ∩ B2)' \ (A1 ∪ A2) .
```

Its elements are the **phantoms**. `join_overshoot` in `src/fca.py` reports
`|Over|` and the normalized `|Over| / |(A1 ∪ A2)''|`.

**Theorem 3.5 (the join strictly overshoots, and it is not fixable inside the
language).** There is a realization in `d = 1` with `|G| = 3`, `|M| = 2` and two
concept extents `A1, A2` such that

1. `A1 ∪ A2` is not a concept extent;
2. `A1 ∪ A2 ≠ G ∩ P` for every `P ∈ L(M)`; indeed `A1 ∪ A2 ≠ G ∩ P` for every
   *convex* `P ⊆ R^1`;
3. the lattice join strictly overshoots: `(A1 ∪ A2)'' ⊋ A1 ∪ A2`.

**[P+V]**

*Proof.* Take `E = (0, 2, 1)^T` (objects `g1, g2, g3` on a line, `g3` between
the other two), `V = (-1, +1)^T`, `tau = (-0.5, 1.5)`. Then attribute `a` holds
of `g` iff `e_g < 0.5` and attribute `b` holds iff `e_g > 1.5`, giving the
incidence table

|      | a | b |
|------|---|---|
| g1   | 1 | 0 |
| g2   | 0 | 1 |
| g3   | 0 | 0 |

Derivations: `{a}' = {g1}`, `{b}' = {g2}`, `{a,b}' = ∅`, `∅' = G`. The four
concepts are `(G, ∅)`, `({g1}, {a})`, `({g2}, {b})`, `(∅, {a,b})`. Take
`A1 = {g1}`, `A2 = {g2}`. Then `B1 ∩ B2 = {a} ∩ {b} = ∅`, so by Proposition 3.3

```
    (A1 ∪ A2)''  =  ∅'  =  G  =  {g1, g2, g3}  ⊋  {g1, g2}  =  A1 ∪ A2 ,
```

giving (3) with `Over = {g3}`. For (1): the extents are exactly
`{G, {g1}, {g2}, ∅}` and `{g1, g2}` is not among them. For (2): `e_{g3} = 1` is
between `e_{g1} = 0` and `e_{g2} = 2`, so any convex `P ⊆ R^1` containing both
`e_{g1}` and `e_{g2}` contains `e_{g3}`; this is Lemma 1.5 in its simplest
instance. ∎

Verified: `check1_minimal.py` reproduces the table via `realize` and reports
`{'union': 2, 'closed': 3, 'phantoms': 1, 'overshoot': 0.333}`.

**Theorem 3.6 (minimality).** `(|G|, |M|) = (3, 2)` is the smallest formal
context, in both parameters simultaneously, admitting a strictly overshooting
join. **[P+V]**

*Proof.* Sufficiency is Theorem 3.5. For necessity:

*`|M| = 1` never overshoots.* The extents are `∅' = G` and `{m}'`. Any union of
two of these is `G` or `{m}'`, both extents.

*`|G| ≤ 2` never overshoots.* The subsets of `G` are `∅, {g1}, {g2}, G` (or
fewer). `G = ∅'` is always an extent, and the union of two extents is either one
of them, or `{g1} ∪ {g2} = G`. Either way it is an extent.

So `|G| ≥ 3` and `|M| ≥ 2` are both necessary. ∎

Exhaustive machine check (`check1_minimal.py`): all `2^{|G|·|M|}` contexts for
`|G| ≤ 4, |M| ≤ 4` with `|G|·|M| ≤ 16`. No overshoot for any `|G| ≤ 2` or any
`|M| = 1`; the first witness at `(3,2)` is the transpose-equivalent of the table
above.

### 3.3 When is the overshoot empty?

**Theorem 3.7 (exact characterization).** `Over(A1, A2) = ∅` iff

```
    (B1 ∩ B2)'  =  B1' ∪ B2'  ,
```

i.e. iff every object satisfying *all* the shared attributes satisfies *all* of
`B1` or *all* of `B2`. Geometrically: iff

```
    G ∩ P_{B1 ∩ B2}  ⊆  P_{B1} ∪ P_{B2} .
```

**[P]**

*Proof.* `A1 ∪ A2 ⊆ (A1 ∪ A2)'' = (B1 ∩ B2)'` always, so emptiness of the
difference is equality. Rewrite via Proposition 3.3 and Theorem 1.3. ∎

**Corollary 3.8 (a cheap sufficient condition for overshoot).** If `B1 ∩ B2 = ∅`
and `A1 ∪ A2 ≠ G`, the join overshoots, with `Over = G \ (A1 ∪ A2)`. **[P]**

*Proof.* `(B1 ∩ B2)' = ∅' = G`. ∎

Corollary 3.8 is the mechanism behind essentially all of the overshoot measured
in §6's pilots: as soon as two concepts share no attribute, their join is the
top of the lattice.

**Proposition 3.9 (the ambient gap, and when it is nonempty).** Set
`C = B1 ∩ B2`, `D1 = B1 \ C`, `D2 = B2 \ C`. The **gap region**
`Γ = P_C \ (P_{B1} ∪ P_{B2})` is nonempty iff `D1 ≠ ∅`, `D2 ≠ ∅`, and there
exist `m1 ∈ D1`, `m2 ∈ D2` such that the system

```
    <v_j, x> > tau_j   (j ∈ C),     <v_{m1}, x> ≤ tau_{m1},     <v_{m2}, x> ≤ tau_{m2}
```

is feasible. **[P+V]**

*Proof.* (⇐) Such an `x` lies in `P_C`, fails `m1` so `x ∉ P_{B1}`, fails `m2`
so `x ∉ P_{B2}`. (⇒) An `x ∈ Γ` lies in `P_C` and outside `P_{B1}`, so it
violates some constraint of `B1`; that constraint cannot be in `C` (it satisfies
those), so it is some `m1 ∈ D1`. Symmetrically for `m2 ∈ D2`. If `D1 = ∅` then
`P_{B1} = P_C` and `Γ = ∅`; likewise `D2`. ∎

This is a finite family of linear programs, so gap-nonemptiness is decidable in
polynomial time. Checked against Monte Carlo on 544 random `(V, tau, B1, B2)`
instances in `d = 2`: 539 agreements; all 5 disagreements were LP-positive /
MC-empty, which is expected for thin or distant gaps and not a contradiction
(`check7_gap.py`).

**Warning 3.10 (the gap can genuinely be empty).** Two proper open convex
subsets *can* cover a convex set. Take `d` arbitrary, `C = ∅`, `m1` with
`H_{m1} = {x_1 < 1}` and `m2` with `H_{m2} = {x_1 > -1}`. Then
`P_{m1} ∪ P_{m2} = R^d = P_C` and `Γ = ∅`. So no argument of the form "a convex
set is never a union of two proper convex subsets" is available; Proposition 3.9
has to be stated as a feasibility condition, not a genericity claim. Verified
(`check7_gap.py`).

**Theorem 3.11 (overshoot is sampling-generic).** Suppose the gap region `Γ` for
a pair of intents has nonempty interior, and objects are drawn i.i.d. from a
distribution `μ` on `R^d` with `μ(Γ) = q > 0`. Then

```
    Pr[ Over(A1, A2) = ∅ ]  ≤  (1 - q)^n  →  0 .
```

**[P]**

*Proof.* If any sampled object lands in `Γ` it lies in `G ∩ P_{B1∩B2}` but in
neither `P_{B1}` nor `P_{B2}`, so by Theorem 3.7 it is a phantom. Independence
gives the bound. ∎

Theorem 3.11 is the bridge between the ambient statement (Proposition 3.9) and
the observable one (Theorem 3.7): **the ambient geometry determines a
probability, the sample determines an outcome.** Verified in miniature: two
orthogonal half-planes with `G` avoiding the third quadrant give zero phantoms;
adding a single object at `(-1,-1)` produces one (`check7_gap.py`).

**Corollary 3.12 (a lower bound with no free parameters).**

```
    Over(A1, A2)  ⊇  ( G ∩ conv(A1 ∪ A2) ) \ (A1 ∪ A2) .
```

**[P]** Immediate from Lemma 1.5. Any object that embeds inside the convex hull
of the union of two extents but belongs to neither is a phantom, in every
realization, in every dimension, for every choice of directions and thresholds.
The bound is loose (Warning 1.7) but it is unconditional.

---

## 4. The trilemma

This is the conceptual payload. Fix the language `L(M)` of a realization and
consider a putative **disjunction operator** `⊔` defined on pairs of concept
extents.

> **Theorem 4.1 (trilemma).** The following three properties cannot hold
> simultaneously:
>
> **(R) Representability.** For every pair of concept extents `A1, A2`, the
> result `A1 ⊔ A2` is again a concept extent — equivalently, `A1 ⊔ A2 = G ∩ P`
> for some `P ∈ L(M)`. (Equivalently again: the result has an intent, so it can
> be fed back into the lattice and composed.)
>
> **(E) Extensional exactness.** For every pair, `A1 ⊔ A2 = A1 ∪ A2` as subsets
> of `G`.
>
> **(U) Universality.** (R) and (E) hold for *every* realization: every
> `d ≥ 1`, every finite `E`, every `V`, every `tau`.
>
> Moreover each of the three pairs {R,E}, {R,U}, {E,U} *is* satisfiable, so none
> of the three legs is redundant. **[P+V]**

*Proof of incompatibility.* Suppose all three. Apply (U) to the realization of
Theorem 3.5 and the pair `A1 = {g1}`, `A2 = {g2}`. By (E),
`A1 ⊔ A2 = {g1, g2}`. By (R), `{g1,g2}` is a concept extent of that context.
But the extents of that context are exactly `{G, {g1}, {g2}, ∅}` (Theorem 3.5),
and `{g1,g2}` is not among them. Contradiction. ∎

*Proof that each pair is satisfiable.*

**(R) ∧ (U), sacrificing (E) — this is the concept lattice.** Define
`A1 ⊔ A2 = (A1 ∪ A2)''`. By Proposition 3.3 this equals `G ∩ P_{B1 ∩ B2}` with
`P_{B1∩B2} ∈ L(M)`, so (R) holds; the definition is uniform in the realization,
so (U) holds. (E) fails on the Theorem 3.5 witness. This is what FCA actually
does, and it is the *only* choice satisfying (R) ∧ (U) that is also sound and
minimal — see Proposition 4.2.

**(E) ∧ (U), sacrificing (R) — this is the set union.** Define
`A1 ⊔ A2 = A1 ∪ A2`. Exact and uniform by construction; (R) fails on the
Theorem 3.5 witness. The cost is that the result has no intent, is not a
polyhedron trace, and cannot be composed: `(A ⊔ B) ⊔ C` is meaningful only if
you leave the lattice, and then `∩` and `⊔` no longer generate a lattice of
concepts. This is exactly the move an LLM would have to make to "represent a
disjunction" — and it is a move *outside* the half-space language, so it cannot
be read off `(V, tau)`.

**(R) ∧ (E), sacrificing (U) — the union-closed realizations.** These exist and
are characterized in Theorem 4.3. Concretely: let all attribute directions be
positive multiples of a single `v ∈ R^d`, `v_m = c_m v` with `c_m > 0`,
thresholds arbitrary. Then the half-spaces `H_m = { x : <v,x> > tau_m / c_m }`
are totally ordered by inclusion, so the rows of `I` are totally ordered by
`⊆`, so the extents form a chain and are trivially closed under union. Set
`A1 ⊔ A2 = A1 ∪ A2`: (R) and (E) both hold. (U) fails because the Theorem 3.5
realization is not of this form. Verified: 400/400 random realizations with
parallel directions were union-closed (`check3_unionclosed.py`). ∎

**Proposition 4.2 (the join is the *unique* best (R)∧(U) operator).** Among
operators satisfying (R), (U), and soundness (`A1 ⊔ A2 ⊇ A1 ∪ A2`), the lattice
join `(A1 ∪ A2)''` is pointwise minimal. **[P]**

*Proof.* Any sound representable result is an extent containing `A1 ∪ A2`, and
`(A1 ∪ A2)''` is by Corollary 1.4 the smallest such. ∎

So the overshoot is not an artifact of a bad choice of join. It is the minimum
price of (R) ∧ (U) ∧ soundness. Nothing cleverer inside the language does
better.

**Theorem 4.3 (exactly which realizations satisfy (R) ∧ (E)).** For a context
`K` define the **row quasi-order** on `G` by `g ≼ h` iff `row(g) ⊆ row(h)`
(object `h` has every attribute `g` has). The following are equivalent:

1. the union of any two concept extents is a concept extent;
2. `A'' = ∪_{g ∈ A} {g}''` for every nonempty `A ⊆ G` (the closure operator is
   *topological* / Alexandrov);
3. the nonempty extents are exactly the nonempty up-sets of `≼`;
4. geometrically: for every nonempty `A ⊆ G`, every object in `G ∩ P_{A'}`
   already lies in `P_{{g}'}` for some single `g ∈ A` — the polyhedron of a set
   is covered, *on `G`*, by the polyhedra of its individual points.

**[P+V]**

*Proof.* First, `{g}'' = { h ∈ G : row(h) ⊇ row(g) }` — the up-set of `g` —
directly from the definitions (`attrs_of({g}) = row(g)`, and
`objs_of(row(g)) = { h : row(h) ⊇ row(g) }`). Verified over 2000 random
contexts, zero violations.

(1 ⇒ 2): `∪_{g∈A}{g}''` is a union of extents, hence an extent by (1) (finite
`A`, iterate), and it contains `A`; so it contains `A''`. Conversely each
`{g}'' ⊆ A''` for `g ∈ A` by monotonicity, so the union is contained in `A''`.

(2 ⇒ 1): for extents `C1, C2`,
`(C1 ∪ C2)'' = ∪_{g ∈ C1∪C2}{g}'' = (∪_{g∈C1}{g}'') ∪ (∪_{g∈C2}{g}'')
= C1'' ∪ C2'' = C1 ∪ C2`.

(2 ⇒ 3): every nonempty extent `A = A'' = ∪_{g∈A}{g}''` is a union of up-sets,
hence an up-set. Conversely if `U` is a nonempty up-set then
`U = ∪_{g∈U}{g}''` (⊆ since `g ∈ {g}''`; ⊇ since `h ∈ {g}''` means `h ⪰ g ∈ U`
so `h ∈ U`), a union of extents, hence an extent by (2 ⇒ 1).

(3 ⇒ 1): up-sets are closed under union.

(4 ⇔ 2): restate (2) with Theorem 1.3 and Corollary 1.4:
`A'' = G ∩ P_{A'}` and `{g}'' = G ∩ P_{{g}'}`. ∎

Verified: 6000 random contexts, `(1) ⇔ (2)` with zero disagreements
(`check3_unionclosed.py`); 4000 random contexts, `(1) ⇔ (3)` with zero
disagreements (`check8_upsets.py`).

**Reading of Theorem 4.3.** (R) ∧ (E) demands that the extent family be an
*Alexandrov topology on `G` induced by a quasi-order*. That is a very strong
structural condition: the geometry must be, on the data, a *hierarchy* — nested
or unrelated, never genuinely crossing. Half-spaces in general position do not
do that; parallel half-spaces do. Which is the point of §5: the hypothesis's own
canonical form pushes the geometry toward general position, i.e. away from the
only regime where its disjunction would be exact.

---

## 5. The dimension budget

### 5.1 What the canonical form costs, and what it buys

Independence of `{v_m}_{m∈M}` requires `m ≤ d`. That is the entire hard cap and
it is worth stating baldly:

**Proposition 5.1.** The paper's canonical form (linearly independent attribute
directions) requires `|M| ≤ d`. **[P]** (Trivial.)

**Theorem 5.2 (independence ⇒ no constraint whatsoever).** Suppose
`rank(V) = m` (so `m ≤ d`). Then for **every** threshold vector `tau ∈ R^m` and
**every** boolean matrix `I ∈ {0,1}^{n×m}` there exist embeddings
`E ∈ R^{n×d}` with `realize(E, V, tau) = Context(I)`. **[P+V]**

*Proof.* `x ↦ Vx` is a surjection `R^d → R^m` because `rank(V) = m`. Fix `g`;
let `s ∈ {+1,-1}^m` with `s_j = +1` iff `I[g,j]`. Choose `e_g` with
`V e_g = tau + s`. Then `<v_j, e_g> - tau_j = s_j`, which is `> 0` exactly on
the prescribed set. ∎

Verified: 500/500 random `7 × 3` contexts realized exactly, `d = 4`, via the
pseudoinverse construction (`check3_unionclosed.py`).

**Corollary 5.3 (the canonical form is unfalsifiable).** In the canonical
regime, the realization hypothesis places **no constraint** on the incidence
matrix, hence none on the concept lattice. Every finite lattice realizable as a
concept lattice on `m` attributes is realizable geometrically with independent
directions. Fitting a canonical-form model to data therefore cannot fail, and a
successful fit is not evidence. **[P]**

**Theorem 5.4 (a second vacuity regime: too few objects).** If `n ≤ d + 1`, then
for **every** boolean `I ∈ {0,1}^{n×m}` there exist `E, V, tau` in `R^d`
realizing it. **[P+V]**

*Proof.* Place the `n` objects at affinely independent points, e.g.
`e_1 = 0, e_i = ε_{i-1}` (standard basis vectors), which is possible since
`n - 1 ≤ d`. For each attribute `j` we need an affine functional
`f(x) = <v_j,x> - tau_j` with prescribed signs at `n` affinely independent
points; the linear system `(E | -1) (v_j, tau_j)^T = y_j` with
`y_j ∈ {+1,-1}^n` has `n` equations in `d+1 ≥ n` unknowns and full row rank
(affine independence), hence a solution. Do this independently per attribute. ∎

Verified: `d ∈ {1,2,3,4}`, `n = d+1`, `m ∈ {1,2,3,5}`, 200 random targets each —
all realized exactly (`check2_geometry.py`). This is the standard "half-spaces
shatter `d+1` affinely independent points" fact; the corresponding upper bound
is Radon's theorem, which gives VC dimension exactly `d+1` for half-spaces in
`R^d`. Checked: for `d ∈ {1,2,3}` and `n = d+2`, 50/50 random point sets failed
to be shattered (`check2_geometry.py`).

> **Methodological consequence for the empirical arms.** A test of the
> hypothesis has content only when **both** `|G| ≥ d + 2` **and** `|M| > d`. In
> any other regime the model fits by construction, and reporting a good fit says
> nothing. This is the single most important operational statement in this
> document.

### 5.2 What `d < |M|` actually forbids

**Theorem 5.5 (row-count bound).** In any realization in `R^d` with `|M| = m`
attributes, the incidence matrix `I` has at most

```
    Φ(m, d)  =  Σ_{k=0}^{d} C(m, k)
```

distinct rows, whatever `n` is. When `d ≥ m` this is `2^m` (no constraint);
when `d < m` it is strictly smaller. **[P+V]**

*Proof.* For `x ∈ R^d` write `r(x) ∈ {0,1}^m` with `r(x)_j = [<v_j,x> > tau_j]`.
The rows of `I` are `r(e_g)`, so it suffices to bound `|r(R^d)|`. Regard
`R = r(R^d)` as a set system on `M`.

*Claim: `R` has VC dimension at most `d`.* Suppose `S ⊆ M` with `|S| = d+1` is
shattered, i.e. all `2^{d+1}` sign patterns on `S` are realized. The `d+1`
normals `{v_j}_{j∈S}` lie in `R^d`, hence are linearly dependent: there is
`λ ≠ 0` with `Σ_{j∈S} λ_j v_j = 0`. Put `f_j(x) = <v_j,x> - tau_j`; then

```
    Σ_{j∈S} λ_j f_j(x)  =  <Σ λ_j v_j , x>  -  Σ λ_j tau_j  =  -c ,   c := Σ λ_j tau_j ,
```

a constant, independent of `x`. Consider two sign patterns:
`P+ : f_j > 0 exactly for j with λ_j > 0`, and
`P- : f_j > 0 exactly for j with λ_j < 0`.
On `P+`, every term `λ_j f_j` is `> 0` (when `λ_j > 0`) or `≥ 0` (when
`λ_j < 0`, since then `f_j ≤ 0`), so `Σ λ_j f_j ≥ 0`, strictly `> 0` if some
`λ_j > 0`. On `P-`, symmetrically, `Σ λ_j f_j ≤ 0`, strictly `< 0` if some
`λ_j < 0`. If `λ` has entries of both signs, `P+` forces `-c > 0` and `P-`
forces `-c < 0`: at least one is infeasible. If `λ > 0` throughout, `P+` forces
`-c > 0` and `P-` (all `f_j ≤ 0`) forces `-c ≤ 0`; again at least one is
infeasible. Symmetrically if `λ < 0` throughout. So `S` is not shattered. ∎(claim)

By Sauer–Shelah, a set system of VC dimension `≤ d` on `m` points has at most
`Σ_{k=0}^{d} C(m,k)` members. ∎

Verified: over 300 random realizations per cell with 400 sampled points,
the bound was respected in all 24 `(m,d)` cells tested, and *attained exactly*
in 13 of them (e.g. `d=2, m=5`: 16 = 16; `d=4, m=4`: 16 = 16) — so the bound is
tight, not merely valid (`check2_geometry.py`).

**Remark 5.6 (the same object counted two ways).** `Φ(m,d)` is also the number
of cells of an arrangement of `m` hyperplanes in general position in `R^d`
(Schläfli; Buck 1943), and `r(R^d)` is exactly the set of cell sign vectors. The
VC-dimension proof above is self-contained and avoids needing general position.
Cover's function-counting theorem is the dual statement on the *object* side:
the number of subsets of `n` points in general position in `R^d` cut out by a
homogeneous hyperplane is `C(n,d) = 2 Σ_{k=0}^{d-1} C(n-1, k)`; with a threshold
(affine separation) the count is `2 Σ_{k=0}^{d} C(n-1,k)`, since affine
separation in `R^d` is homogeneous separation in `R^{d+1}`.

**Corollary 5.7 (bound on distinct attribute extents).** The number of distinct
extents of the form `{m}'`, `m ∈ M`, is at most `2 Σ_{k=0}^{d} C(n-1,k)`
regardless of `|M|`. Since every extent is an intersection of such extents
(Theorem 2.2), the whole lattice is generated by at most that many
half-space traces. **[P]** (Cover's bound applied to `E` and the family of
attribute half-spaces.)

**Warning 5.8 (`Φ` is *not* the binding constraint at LLM scale — do not
oversell it).** For `d = 4096` and `m = 10^7`, `log2 Φ(m,d) ≈ 5.2 × 10^4`, so
the cap on distinct rows is about `2^{52005}` — astronomically more than any
conceivable `|G|`. Theorem 5.5 bites in the low-`d` experimental regime
(`d ≲ 10`), not in the LLM regime. Numbers in `check6_budget.py`. The genuinely
binding constraint at LLM scale is §5.3.

### 5.3 Which boolean matrices are realizable at all: sign-rank

**Definition 5.9.** For a sign matrix `S ∈ {±1}^{n×m}`,
`sign-rank(S) = min{ rank(R) : R real, sign(R) = S }`. It is exactly the minimum
dimension in which `S` is realizable by half-spaces *through the origin*
(Alon–Moran–Yehudayoff, *Sign rank versus VC dimension*, arXiv:1503.07648, §1).

**Proposition 5.10 (realizability = sign-rank, up to one dimension).** Let
`S = 2I - 1`. Then

```
    sign-rank(S) - 1   ≤   d_min(I)   ≤   sign-rank(S) ,
```

where `d_min(I)` is the least `d` in which `I` is realizable in the sense of
Definition 1.1. **[P/sk]**

*Proof sketch.* Ties (`<v_m,e_g> = tau_m`) can be removed without changing `I`
by raising each `tau_m` slightly, exactly as in Proposition 1.8 — `G` is finite
— so WLOG all entries of `R = E V^T - 1 tau^T` are nonzero and `sign(R) = S`.
Then `rank(R) ≤ rank(EV^T) + 1 ≤ d + 1`, giving `sign-rank(S) ≤ d + 1`, i.e. the
left inequality. For the right: `sign-rank(S) = k` gives a factorization
`R = XY^T` with `X ∈ R^{n×k}`, `Y ∈ R^{m×k}`, so setting `E = X`, `V = Y`,
`tau = 0` realizes `I` in `d = k`. **Gap I did not close:** the two bounds
differ by one because a rank-`(d+1)` factorization need not admit a basis change
putting an all-ones column into `X` (which is what an *affine* realization in
`R^d` requires). I do not know an example where the left inequality is strict,
and I did not search for one. **[X]** on the exact value.

**Theorem 5.11 (counting: almost no context is realizable once `n, m ≫ d`).**
The number of `n × m` boolean matrices realizable in `R^d` is at most

```
    ( 8e · nm / ℓ )^ℓ ,      ℓ = (n + m) d + m ,
```

whenever `nm ≥ ℓ`. **[P]**

*Proof.* The `nm` quantities `f_{gj} = <e_g, v_j> - tau_j` are polynomials of
degree 2 in the `ℓ = nd + md + m` real parameters `(E, V, tau)`. Warren's
theorem (H. E. Warren, *Lower bounds for approximation by nonlinear manifolds*,
Trans. Amer. Math. Soc. **133** (1968), 167–178) bounds the number of sign
patterns of `N` polynomials of degree `≤ k` in `ℓ` variables, for `N ≥ ℓ`, by
`(4ekN/ℓ)^ℓ`. Put `N = nm`, `k = 2`. Each realizable `I` arises from at least
one sign pattern. ∎

This is the rectangular version of Alon–Moran–Yehudayoff's Lemma 22 (`N × N`
sign matrices of sign rank `≤ r` number at most `(O(N/r))^{2Nr} ≤
2^{O(rN log N)}`), which is itself the Alon–Frankl–Rödl argument.

**Corollary 5.12 (quantitative threshold).** Let `K*(d)` be the least `K` such
that the bound of Theorem 5.11 with `n = m = K` is below `2^{K^2}`. Then for
`K ≥ K*(d)` the fraction of `K × K` contexts realizable in `R^d` is at most
`2^{-Ω(K^2)}`. Numerically (`check6_budget.py`):

| `d` | `K*(d)` | `K*(d)/d` | log₂(realizable fraction) at `K = 2K*` |
|---:|---:|---:|---:|
| 8 | 125 | 15.6 | −0.434 · K² |
| 256 | 3,752 | 14.7 | −0.432 · K² |
| 1,024 | 14,985 | 14.6 | −0.432 · K² |
| 2,048 | 29,963 | 14.6 | −0.432 · K² |
| **4,096** | **59,918** | **14.6** | **−0.432 · K²** |
| 12,288 | 179,737 | 14.6 | −0.432 · K² |

`K*(d) ≈ 14.6 d`. **[V]** on the arithmetic; **[P]** on the underlying bound.

### 5.4 The LLM argument, made quantitative

Take `d = 4096` (a mid-size residual stream; the argument scales linearly in
`d`, see the table). Two independent verdicts.

**(a) The canonical form fails by three to four orders of magnitude.** Exact
linear independence of attribute directions caps `|M| ≤ d = 4096`. Published
sparse-autoencoder decompositions of production models report far more
interpretable features than that: 34 million on Claude 3 Sonnet's middle-layer
residual stream (Templeton et al., *Scaling Monosemanticity*, arXiv:2605.29358 —
abstract verified), 16 million latents on GPT-4 (Gao et al., *Scaling and
evaluating sparse autoencoders*, arXiv:2406.04093 — abstract verified). Even
granting that SAE latents are a generous proxy for FCA attributes, the ratio
`|M|/d` is `10^3`–`10^4`. **The canonical form cannot hold globally.** [P]

**(b) But the failure is graceful, and one must not overstate it.** Near-
independence survives far past `m = d`. The Welch bound forces mutual coherence
`μ ≥ sqrt((m-d)/(d(m-1)))`, which for `d = 4096` is only `0.0111` at
`m = 8192` and `0.0156` at `m = 10^7` — essentially the `1/sqrt(d)` floor
(`check6_budget.py`). And `ε`-orthogonal packings of size `exp(c ε² d)` exist by
the Johnson–Lindenstrauss construction, so `4096` dimensions comfortably admit
on the order of `2^{30}` directions at coherence `0.1`. **So "d = 4096 cannot
support more than 4096 concepts" is FALSE and should not be argued.** The
correct statement is the one in (a) — *exact* independence fails — plus (c).

**(c) The binding constraint is sign-rank, and it bites around `|G| ≈ 6 × 10^4`.**
By Corollary 5.12, for `d = 4096` a *generic* square context stops being
realizable at about `K* ≈ 6 × 10^4` objects and attributes; beyond that, the
realizable contexts are a `2^{-Ω(K²)}` fraction. Concretely at `d = 4096`
(`check6_budget.py`):

| `n` | `m` | log₂(#realizable) | `nm` | verdict |
|---:|---:|---:|---:|---|
| 10³ | 10³ | 10⁶ | 10⁶ | bound vacuous — everything may be realizable |
| 10⁴ | 10⁴ | 3.9 × 10⁸ | 10⁸ | bound vacuous |
| 5×10⁴ | 5×10⁴ | 2.9 × 10⁹ | 2.5 × 10⁹ | bound vacuous (just) |
| 10⁵ | 10⁵ | 6.6 × 10⁹ | 10¹⁰ | fraction ≤ 2^(−3.4×10⁹) |
| 5×10⁴ | 10⁶ | 3.4 × 10¹⁰ | 5 × 10¹⁰ | fraction ≤ 2^(−1.6×10¹⁰) |
| 10⁵ | 10⁷ | 3.7 × 10¹¹ | 10¹² | fraction ≤ 2^(−6.3×10¹¹) |

So: at SAE-realistic attribute counts and any object set past a few tens of
thousands, the set of contexts a `4096`-dimensional geometry can realize is a
vanishing fraction of all contexts. If the LLM's actual attribute structure were
"generic", the hypothesis would be false with overwhelming probability. The
hypothesis is therefore a substantive claim that the structure is *very* far
from generic — and that is the claim the empirical arms should be testing, not
whether some lattice can be fitted.

**Caveat 5.13.** The `2^{-Ω(K²)}` statement is a counting argument about the
uniform measure on boolean matrices. Real linguistic contexts are of course not
uniform. Counting cannot show a *specific* context is unrealizable; it shows
that realizability is a strong structural hypothesis, not a formality. To show a
specific context unrealizable you need a sign-rank lower bound, and no explicit
matrix with sign-rank much above `sqrt(N)` is known (Alon–Moran–Yehudayoff §1.3).
**[X]** — I could not produce a concrete unrealizability certificate for any
LLM-derived context.

### 5.5 The independence–content tension, stated once

Combining Theorem 5.2 and Theorem 5.5:

> **Theorem 5.14.** Let `r = rank(V)`.
> - If `r = m` (canonical form) then *every* incidence matrix is realizable with
>   those directions: the hypothesis constrains nothing.
> - If `r < m` then at most `Φ(m, r) < 2^m` of the `2^m` possible attribute
>   profiles can occur: the hypothesis constrains the data.
>
> Hence **the canonical form holds exactly when the hypothesis has no empirical
> content, and the hypothesis has content exactly when the canonical form
> fails.** **[P]**

And by §4, the same dial runs the other way for disjunction: general-position
directions (low coherence) maximize overshoot, while the exact-disjunction
regime of Theorem 4.3 requires the extents to be a hierarchy — which parallel,
maximally *dependent* directions produce. The paper's canonical form is
simultaneously the least testable and the worst-behaved regime for the join.

---

## 6. Predictions

All predictions are about the *realized* context, so the empirical arms can
compute them from `realize(...)` and `join_overshoot(...)` with no extra
machinery. Metric: mean phantom fraction `|Over| / |(A1 ∪ A2)''|` over pairs of
distinct concepts, and the fraction of pairs with `|Over| > 0`.

Pilot numbers below are from `check5_pilots2.py` (`n = 300`, extents sampled by
random attribute subsets to avoid the small-intent bias of enumerating `B` in
size order; 1500 random pairs per realization; 10 realizations per cell). They
are pilots, not results — the empirical arm owns the real measurement.

**P1 — Overshoot rises with `d` and saturates at `d ≈ |M|`.** Because the
arrangement gains cells (Theorem 5.5) until `d ≥ m`, at which point all `2^m`
profiles are available (Theorem 5.2) and the context is unconstrained.
*Falsifier:* overshoot flat or decreasing in `d` over `1 ≤ d ≤ 2m`.
Pilot (`m = 12`, `p = 0.5`): mean overshoot `0.00` at `d=1`, `0.19` at `d=2`,
`0.34` at `d=3`, `0.44` at `d=5`, then flat `0.45–0.47` for `d ∈ [8, 256]`.
**[V]**

*Sub-prediction P1a:* the `d = 1` zero is an artifact of equal per-attribute
densities (all rays coincide). With heterogeneous densities `p_j ~ U(0.1, 0.9)`,
`d = 1` gives `0.08` — small but nonzero, consistent with Theorem 3.5 which
exhibits overshoot in `d = 1`. Empirical arms must not report "`d=1` has no
overshoot"; that is a design artifact. **[V]**

**P2 — Overshoot *decreases* monotonically with attribute-direction coherence.**
This is the counterintuitive one and the sharpest test of §5.5. High coherence
pushes the half-spaces toward nesting, i.e. toward the Alexandrov regime of
Theorem 4.3 where unions are already closed.
*Falsifier:* overshoot increasing in coherence, or flat.
Pilot (`n=300`, `m=12`, `d=32`, `p=0.5`), sweeping an interpolation parameter
`α` toward a common direction:

| coherence | 0.42 | 0.51 | 0.58 | 0.72 | 0.84 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| mean overshoot | 0.455 | 0.390 | 0.278 | 0.193 | 0.141 | 0.096 | 0.076 |
| frac. strict | 0.79 | 0.76 | 0.72 | 0.71 | 0.70 | 0.66 | 0.58 |

Monotone over more than a `6×` range. **[V]** *Caveat:* my extent sampler caps
at 120 distinct extents, so at `α = 0.999` the concept count itself collapses
and the last point is confounded; the empirical arm should enumerate concepts
exactly and control for concept count.

**P3 — Overshoot decreases monotonically with attribute density `p`.**
Mechanism: Corollary 3.8. Low density makes intents small and disjoint, so joins
collapse to the top of the lattice.
*Falsifier:* non-monotone in `p` over `[0.1, 0.9]`, or increasing.
Pilot (`n=300`, `m=12`, `d=32`): `0.70` at `p=0.10`, `0.68` at `0.20`, `0.60` at
`0.35`, `0.46` at `0.50`, `0.29` at `0.65`, `0.13` at `0.80`, `0.013` at `0.95`.
Note the turn near `p ≈ 0.10`: at the very sparse end the concept count also
collapses (40 extents versus 120), so the extreme-left point is confounded the
same way. **[V]**

**P4 — Overshoot rises with `|M|/d` and saturates near `1`.** Same mechanism as
P1 seen from the other side.
Pilot (`n=300`, `d=32`, `p=0.5`): `0.04` at `m=2`, `0.16` at `m=4`, `0.37` at
`m=8`, `0.52` at `m=16`, `0.53` at `m=32`, `0.53` at `m=64`, `0.50` at `m=128`.
**[V]**

**P5 — The hull lower bound holds exactly, and is loose by roughly an order of
magnitude.** For every pair, `(G ∩ conv(A1 ∪ A2)) \ (A1 ∪ A2) ⊆ Over(A1, A2)`.
*Falsifier:* any single violation — that would refute Lemma 1.5 and hence the
whole half-space model, not just the join claim. This is a **calibration gate**,
not a measurement: it must pass.
Pilot: 0 violations in 4465 cases; hull-explained phantoms were 85 of 723
(≈12%). **[P** for the containment, **V** for the looseness ratio.**]**

**P6 — Meet exactness is invariant.** `A1 ∩ A2` is always closed, in every
realization, at every `d`, every coherence, every density. Any sweep of the meet
should be a flat line at zero error. If it is not, the pipeline is broken.
This is `assert_meet_closed`. **[P]**

**P7 — Vacuity regimes must show a step change.** With `n ≤ d + 1`, or with
`m ≤ d` and `V` full-rank, the realization can fit any target context (Theorems
5.2, 5.4). So a "can we fit the observed lattice?" experiment should succeed
with probability 1 in those regimes and start failing only past
`n ≥ d + 2` and `m > d`. If a fitting procedure fails inside the vacuity regime,
the failure is optimization, not geometry.
*Concrete check:* sample a random target `I` with `n = d+1`; the construction in
Theorem 5.4 realizes it exactly. Verified for `d ≤ 4`. **[P+V]**

**P8 — Overshoot is sampling-driven, and saturates below 1.** Increasing `n` by
sampling more objects from the same distribution increases overshoot, because
each additional object is another chance to land in some pair's gap region
(Theorem 3.11). But it saturates strictly below 1: pairs whose gap region is
*empty* (Warning 3.10, Proposition 3.9) can never overshoot no matter how many
objects are drawn, and those pairs are a constant fraction of the lattice.
*Falsifier:* overshoot decreasing in `n`, or converging to 1.
Pilot (`m=10`, `d=16`, `p=0.5`, unbiased extent sampling, `check9_n.py`): mean
overshoot `0.286 → 0.354 → 0.398 → 0.409 → 0.420 → 0.416` for
`n = 20, 50, 100, 400, 1000, 4000`; fraction of strictly overshooting pairs
`0.70 → 0.76 → 0.77 → 0.75 → 0.76 → 0.74` — rising then flat at ≈ 0.75, not
approaching 1, exactly as the empty-gap argument predicts. **[V]**

**P9 (conjecture, offered for falsification).** For random Gaussian `E, V` with
per-attribute density `p` and `d ≥ m`, the expected phantom fraction converges
as `n → ∞` to a limit depending only on `(m, p)`, not on `d`. The pilot's flat
`0.45–0.47` across `d ∈ [8, 256]` at fixed `m = 12` is consistent with this.
I have no proof and no candidate closed form. **[C]**

---

## 7. Numerical verification log

Scripts live in `/tmp/lrh-theory/` (throwaway, not part of the experiment
directory) and import `src/fca.py`.

| Script | Checked | Result |
|---|---|---|
| `check1_minimal.py` | All `2^{nm}` contexts, `n,m ≤ 4`, `nm ≤ 16`, for strict join overshoot; the `d=1` realization of the minimal witness | Minimality of `(3,2)` confirmed; `realize` reproduces the target table exactly; phantom fraction `1/3` |
| `check2_geometry.py` | Hull lemma (Lemma 1.5); hull-bound looseness; row bound `Φ(m,d)` (Theorem 5.5) over 24 `(m,d)` cells; `n ≤ d+1` shattering (Theorem 5.4); non-shattering at `n = d+2` | 0/4465 lemma violations; 638 of 723 phantoms outside the hull; bound respected in all cells and attained in 13; 800/800 targets realized at `n=d+1`; 0/150 full shatterings at `n=d+2` |
| `check3_unionclosed.py` | Alexandrov equivalence (Thm 4.3, 1⇔2) on 6000 contexts; `{g}''` = up-set on 2000 contexts; parallel directions ⇒ union-closed; independence ⇒ any `I` realizable (Thm 5.2) | 0 disagreements; 0 violations; 0/400 failures; 0/500 failures |
| `check4_pilots.py` | First-pass sweeps | **Superseded** — its extent enumerator walked attribute subsets in size order and so was biased toward small intents. Nothing in §6 is quoted from it. |
| `check9_n.py` | P8 (`n`-sweep) redone with unbiased extent sampling | Numbers quoted in P8; overshoot saturates near 0.75, not 1 |
| `check5_pilots2.py` | P1–P4 sweeps with unbiased extent sampling and heterogeneous densities | Numbers quoted in §6 |
| `check6_budget.py` | Warren threshold `K*(d)`; realizable-fraction table at `d=4096`; `log2 Φ` at LLM scale; Welch bound; the (T3) ambient-vs-trace witness | `K* ≈ 14.6 d`; tables in §5.3–5.4; `(0.1,0.1)` separates `P_{a,b}` from `P_{a,b,c}` with identical traces |
| `check7_gap.py` | LP characterization of the ambient gap (Prop 3.9) vs Monte Carlo; the two-half-planes empty-gap witness (Warning 3.10); ambient-gap-nonempty but trace-overshoot-empty | 539/544 agree, all 5 disagreements LP-positive/MC-empty (expected); empty gap confirmed; 0 phantoms until one object is placed in the gap, then 1 |
| `check8_upsets.py` | Theorem 4.3 (1 ⇔ 3) on 4000 contexts | 0 disagreements |

**Proved without numerics:** Theorems 1.3, 2.1, 2.2, 3.6 (necessity), 3.7, 3.11,
4.1, 4.2, 5.11; Propositions 1.8, 3.1, 5.1; Corollaries 1.4, 1.6, 3.8, 3.12,
5.3, 5.7, 5.12 (bound), 5.14.

**Verified numerically only (no proof here):** all §6 pilot magnitudes and
monotonicities (P1–P4, P8, P9); the looseness ratio in Warning 1.7; the tightness
of `Φ(m,d)` in specific cells; the arithmetic in Corollary 5.12's table.

---

## 8. What I could not establish

Listed so no downstream arm treats these as settled.

1. **Exact `d_min` versus sign-rank.** Proposition 5.10 leaves a one-dimension
   gap. The obstruction is whether a rank-`(d+1)` sign-factorization can always
   be transformed to put an all-ones column in the object factor (which is what
   an affine realization needs). I did not resolve it, and I did not search for
   a counterexample. **[X]**

2. **A concrete unrealizability certificate.** Theorem 5.11 is a counting
   argument: it shows almost all contexts are unrealizable at `d = 4096` past
   `|G| ≈ 6 × 10^4`, but exhibits none. Explicit sign-rank lower bounds above
   `sqrt(N)` are a known open problem, so I do not expect to fix this. **[X]**

3. **A Radon- or Tverberg-based forcing theorem for overshoot.** I wanted "if
   `|G| ≥ d + 2` then some pair of extents must overshoot." I could not prove
   it and I believe it is **false**: `n` points can be in convex position in
   `R^d` for any `n`, so no object need lie in any hull, and Corollary 3.12 gives
   nothing. Radon does bound the VC dimension of half-spaces at `d+1`
   (Theorem 5.4's converse), but that constrains shattering, not overshoot. The
   honest forcing statement is the probabilistic one, Theorem 3.11. **[X]**

4. **A testable consequence of Helly.** (T5) is a real constraint on the
   *ambient* polyhedra — no minimal geometrically infeasible attribute set
   exceeds `d+1` — but `B' = ∅` does not imply `P_B = ∅`, so the observable
   context never reveals ambient infeasibility. I could not construct a test.
   **[X]**

5. **Closed form for the asymptotic phantom fraction.** P9 is a conjecture with
   no candidate expression.

6. **Anything about the paper's WordNet validation.** I did not read the
   experimental section of arXiv:2603.01227 — only its abstract, which I fetched
   and verified. Whether their empirical claims survive these arguments is for
   the empirical arms.

7. **Whether SAE latents are the right proxy for FCA attributes.** §5.4(a)'s
   three-orders-of-magnitude argument depends on that identification. It is a
   plausible reading of the Linear Representation Hypothesis, not a theorem, and
   a defender could reasonably insist that the "attributes" of a lattice
   representation are a much smaller curated set. If `|M| ≤ 4096` after such
   curation, argument (a) evaporates — though (c) and Theorem 5.14 do not, since
   Theorem 5.14 applies at every scale. **[X]**

---

## 9. References

Verified by fetch or search during the writing of this document; none cited from
memory.

- R. Wille and B. Ganter. *Formal Concept Analysis: Mathematical Foundations.*
  Springer, 1999. (Notation and the Basic Theorem; the source `src/fca.py`
  follows.)
- Bo Xiong. *The Lattice Representation Hypothesis of Large Language Models.*
  arXiv:2603.01227. (Abstract fetched and confirmed to state that "linear
  attribute directions with separating thresholds induce a concept lattice",
  validated against WordNet.)
- T. M. Cover. *Geometrical and statistical properties of systems of linear
  inequalities with applications in pattern recognition.* IEEE Trans. Electronic
  Computers **EC-14**(3):326–334, 1965. Counting function
  `C(N,d) = 2 Σ_{k=0}^{d-1} C(N-1,k)` for homogeneous separation of points in
  general position.
- R. C. Buck. *Partition of space.* Amer. Math. Monthly, 1943. (Region count of
  a general-position hyperplane arrangement; classically Schläfli.)
- H. E. Warren. *Lower bounds for approximation by nonlinear manifolds.*
  Trans. Amer. Math. Soc. **133**:167–178, 1968. Sign-pattern bound
  `(4ekN/ℓ)^ℓ` for `N ≥ ℓ` polynomials of degree `≤ k` in `ℓ` variables.
- N. Alon, S. Moran, A. Yehudayoff. *Sign rank versus VC dimension.*
  arXiv:1503.07648. Definition of sign-rank as `min{rank(M) : sign(M) = S}` and
  its identification with minimum dimension of a homogeneous half-space
  embedding; Theorem 21 (Warren) and Lemma 22 (`N × N` sign matrices of sign
  rank `≤ r` number at most `(O(N/r))^{2Nr} ≤ 2^{O(rN log N)}`).
- N. Alon, P. Frankl, V. Rödl. First linear lower bounds on the sign-rank of
  random sign matrices (as reported and used in Alon–Moran–Yehudayoff §1.3).
- Radon's theorem and the VC dimension of half-spaces in `R^d` being exactly
  `d+1`; Helly's theorem. Standard; statement and the Radon-based upper-bound
  proof confirmed by search.
- L. L. Welch. Coherence bound `μ ≥ sqrt((m-d)/(d(m-1)))` for `m` unit vectors
  in `d` dimensions. Statement confirmed by search.
- Johnson–Lindenstrauss packing: `exp(c ε² d)` unit vectors with pairwise inner
  products in `[-ε, ε]` fit in `R^d`. Statement confirmed by search.
- A. Templeton et al. *Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet.* arXiv:2605.29358. Abstract fetched: "We
  trained sparse autoencoders with up to 34 million features on the model's
  middle layer residual stream."
- L. Gao et al. *Scaling and evaluating sparse autoencoders.* arXiv:2406.04093
  (6 June 2024). Abstract fetched: "we train a 16 million latent autoencoder on
  GPT-4 activations for 40 billion tokens."
