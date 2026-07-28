# The Erdős–Gyárfás conjecture

> Every finite graph with minimum degree at least 3 contains a cycle whose length
> is a power of two.

Posed in 1995. Erdős expected it to be **false** and offered $100 for a proof,
$50 for a counterexample. Still open.

A full write-up with proofs and figures is in [`note.html`](note.html).
This README is the summary, split by what actually holds up.

---

## The reformulation

A graph on `n` vertices cannot contain a cycle longer than `n`, so a counterexample
only has to dodge the powers of two that are **at most** `n` — every larger one is
absent for free. Following Exoo, let

> **f(k)** = the smallest order of a cubic graph with no cycle of length 2^m for any m ≤ k.

**Proposition 1.** The conjecture is false for cubic graphs **iff** `f(k) ≤ 2^(k+1) − 1`
for some k ≥ 2.

*Proof.* If `f(k) = N ≤ 2^(k+1) − 1`, a witness on `N` vertices avoids 4, …, 2^k, and
since `N < 2^(k+1)` those are all the powers of two it could contain — so it is a
counterexample. Conversely a cubic counterexample on `n` vertices, with
`k = ⌊log₂ n⌋`, gives `f(k) ≤ n ≤ 2^(k+1) − 1`. ∎

So the conjecture is a race between two integer sequences:

| k | f(k) | threshold 2^(k+1) | margin | log₂f − k |
|---|---|---|---|---|
| 2 | 10 | 8 | +2 | 1.322 |
| 3 | 24 | 16 | +8 | 1.585 |
| 4 | ≥54 | 32 | +22 | ≥1.755 |
| 5 | ? (≤450, disputed — see below) | 64 | ? | ? |

The normalised gap is increasing, and both known growth ratios (2.40, ≥2.25) exceed 2.

**Proposition 2.** If `f(k+1) ≥ 2·f(k)` for all k ≥ 2, the conjecture holds for cubic
graphs — since then `f(k) ≥ 10·2^(k−2) = 2.5·2^k > 2^(k+1) − 1`.

This turns a statement quantified over all graphs into a monotonicity property of one
integer sequence: *avoiding one more power of two must at least double the order
required.* **No general lower bound on f(k) has ever been published**; that absence is
the real state of the problem.

## Why counting cannot settle it

**Proposition 3.** No lower bound on the *number* of distinct cycle lengths can prove
the conjecture.

A counterexample on `n` vertices must dodge `⌊log₂ n⌋ − 1` values out of the `n − 2`
available. At n = 10⁶ that is 18 values out of a million — 0.0018%. A graph realising
all but a vanishing fraction of lengths can still miss precisely the powers of two, so
cardinality never separates the two cases.

This rules out the whole "many cycle lengths" line — Bondy–Vince, Fan, Liu–Ma,
Gao–Huo–Liu–Ma, and Bucić–Gishboliner–Sudakov's `n^(1−o(1))` bound for Hamiltonian
graphs — as a route to a proof. Erdős–Gyárfás is not a density statement; it needs a
mechanism producing a cycle of *prescribed* length. At minimum degree exactly 3 the
consecutive-lengths machinery yields two near-consecutive lengths, against the
~log₂ n needed at exponentially spaced positions.

## Arithmetic constructions are all blocked

**Proposition 4.** For any m ≥ 3, no graph with minimum degree ≥ 3 has all its cycle
lengths in one residue class mod m. In particular every such graph has a cycle of
length not divisible by 3.

*Proof.* Bondy–Vince (1998): every simple graph with at most two vertices of degree
less than three, other than K₁ and K₂, has two cycles whose lengths differ by one or
two. Two lengths in the same class mod m differ by a multiple of m ≥ 3. ∎

This kills the most natural construction — since 2^k mod 3 ∈ {1,2}, a graph with all
cycle lengths divisible by 3 would dodge every power of two at once. It pairs with
Chen–Saito, which gives the complementary half.

The odd-cycle route dies too: a theta subgraph with paths of lengths a, b, c has cycles
summing to 2(a+b+c), so they cannot all be odd; hence every block of an even-cycle-free
graph is an edge or an odd cycle, and minimum degree drops to 2.

## A counterexample must straddle

**Proposition 5** (for 3-connected cubic graphs). For k ≥ 3, no counterexample has its
spectrum inside a single gap (2^k, 2^(k+1)). Fitting inside forces girth > 2^k, which
forces many vertices via the Moore bound, which forces a long cycle via
Liu–Yu–Zhang's Ω(n^0.8) circumference bound — back out of the gap. The Moore bound is
doubly exponential in k while the gap ceiling is only singly exponential, so the
mismatch explodes: closed by a factor of 10⁶ by k = 6.

The k = 2 case needs a direct check (the asymptotic bound is too weak there): over all
26,731 connected graphs with δ ≥ 3 and girth ≥ 5 on up to 17 vertices, none has
circumference ≤ 7.

**The 3-connectivity hypothesis is load-bearing.** Chain t copies of K₄ with single
bridges: minimum degree 3, arbitrarily many vertices, circumference exactly 4.

## Sharpening Carr (2026): 4/7 → 2/3

Carr ([arXiv:2605.22844](https://arxiv.org/abs/2605.22844)) proves that a *minimal*
counterexample satisfies (1) every vertex is adjacent to a vertex of degree exactly 3,
and (2) the vertices of degree ≥ 4 form an independent set — then concludes at least
4/7 of vertices are cubic. That proof uses only (2).

**Proposition 6.** At least **2/3** of the vertices are cubic.

*Proof.* By (1) each v ∈ V₃ has a neighbour inside V₃, so it sends at most 2 of its 3
edges out: `e(V₃,V₄₊) ≤ 2|V₃|`. By (2), V₄₊ is independent and each of its vertices has
degree ≥ 4, all into V₃: `e(V₃,V₄₊) ≥ 4|V₄₊|`. So `|V₄₊| ≤ |V₃|/2` and
`n ≤ (3/2)|V₃|`. ∎

Tight: 8 cubic vertices in a perfect matching plus 4 of degree 4, each cubic vertex
joined to two of the latter — 8/12 exactly.

Carr's Lemma 0.1 also says a minimal counterexample has no proper subgraph of minimum
degree 3. Narins–Pokrovskiy–Szabó ([arXiv:1408.5289](https://arxiv.org/abs/1408.5289))
note any graph with ≥ 2n−1 edges has such a subgraph, giving **e(G) ≤ 2n−2**. Their
"degree 3-critical" class fixes e = 2n−2 and forbids proper *induced* subgraphs, so the
Bollobás–Brightwell circumference results do not transfer.

## A gap in Exoo's f(5) ≤ 450

[`src/tutte_coxeter_lemma.py`](src/tutte_coxeter_lemma.py) — run it.

Exoo ([arXiv:1403.5636](https://arxiv.org/abs/1403.5636)) justifies G₄₅₀ having no
32-cycle in one sentence: *"any 8-cycle in Tutte-Coxeter contains (at least) two
consecutive edges on the outer Hamiltonian cycle."*

**That is false.** Tutte–Coxeter has 90 distinct 8-cycles; **10 of them (11%)** use no
two consecutive outer edges, alternating strictly between outer edges and chords. So
the stated argument has a gap.

Whether the *conclusion* also fails is not settled here. A reconstruction of G₄₅₀ from
the paper's TikZ source does contain a 32-cycle projecting onto one of those ten, but
that rests on the reconstruction being faithful, and Exoo's construction data is not in
the public record — his pages carry drawings only, and an Internet Archive listing of
his former host shows no graph6 files. His site does state, in a page predating the
paper, that the smallest graph he knew of with no 4-, 8-, 16- or 32-cycle had **540**
vertices; that is the fallback if 450 does not stand.

This does not disturb anything above — the propositions use *lower* bounds on f, so a
failing upper bound puts the sequence further ahead.

---

## Computation

Everything here is `snarkhunter` piped into [`src/filt.c`](src/filt.c), an exact
cycle-length test: for target length L, DFS from each vertex over paths visiting only
larger vertices, so each cycle is found once from its minimum vertex, exiting on the
first hit.

**Tooling decided what was reachable.** `nauty-geng` asked for cubic graphs enumerates
everything and filters to 3-regular — ~145 graphs/sec/core, generation-bound, with the
cycle test idling under 1% CPU. `snarkhunter` constructs them directly: **385,000/sec**
through the same pipeline, a factor of 2,600. Order 24 goes from "never finished" to
295 seconds.

| order | all connected cubic | no C₄ and no C₈ | contains C₁₆ | counterexamples |
|---|---|---|---|---|
| 24 | 117,940,535 | 4 | all 4 | 0 |
| 26 | 2,094,480,864 | 23 | all 23 | 0 |

Both counts match OEIS A002851 exactly, so the censuses are complete rather than
sampled. **Both reproduce Markström (2004) rather than extending it** — his Table 3
gives 4, 23 and 251 at orders 24, 26 and 28. Canonically labelling with `nauty-labelg`,
his graphs and these agree **graph-for-graph, zero differences either way** at both
orders. That is worth having as independent verification of a computation whose source
paper is hard to obtain, and nothing more.

**Girth is a trap.** Restricting to girth ≥ 5 shrinks the order-26 search 60-fold and
returns nothing — but 3 is not a power of two, so triangles are legal, and *every one*
of the 27 extremal graphs at orders 24 and 26 has girth 3. The shortcut would have
looked like a clean result.

### An order-26 near-miss

`data/near_miss_order26.g6` — cubic, 26 vertices, spectrum **{3, 5, 6, 7, 9, …, 26}**:
every length from 3 to 26 except exactly 4 and 8. It has one triangle; contract it and
what remains is a cubic graph on 24 vertices of girth 5 whose spectrum is the full run
{5,…,24}, including 8 and 16.

The 24-vertex graph has **exactly nine 8-cycles, and all nine pass through the
contracted vertex.** Blowing that vertex into a triangle moves every one of them to
length 9 and leaves nothing at 8; its ten 7-cycles all avoid that vertex, so they stay
at 7. One triangle, placed at the single vertex where it annihilates a whole cycle
length — Markström's and Exoo's gadget method in its smallest form.

It also sharpens Proposition 3: the graph realises 22 of its 24 available lengths.
Dodging powers of two is not about a sparse spectrum; it is a local operation shifting
one length off a target.

---

## Where it is actually open

Markström settled every cubic order below 29 in his paper, and his data page records a
complete search of all cubic graphs on **N ≤ 52** for {4, 8, 16}-avoidance with none
found — the source of both the "≥30 vertices" bound and the f(4) ≥ 54 that Exoo could
only cite as unpublished.

So the first untested case is five orders wide:

> **A cubic counterexample needs 54 ≤ n ≤ 62, even, avoiding cycles of length
> 4, 8, 16 and 32.**

f(5) in full is not required — nothing can hide above 63, since at n ≥ 64 the graph
must also dodge 64 and the question moves to k = 6. Brute force is hopeless at that
size; Markström reached 52 with a modified `minibaum` pruning on cycle length during
generation. Extending that by five orders is the concrete next step.

---

## Files

```
src/tutte_coxeter_lemma.py   the refutation above — self-contained, run it
src/filt.c                   exact cycle-length filter (gcc -O3 -march=native)
src/window_search.py         girth/circumference search for Proposition 5
src/exoo_graphs.py           H7/H15 gadgets and G78/G420/G450, from the paper's TikZ
src/existence.py             cycle-existence tests used by the reconstruction
src/validate.py              ground-truth regression suite for the above
data/order24_mine.g6         4 graphs, this session
data/order24_markstrom.g6    4 graphs, Markström's file — identical up to isomorphism
data/order26_mine.g6         23 graphs, this session
data/order26_markstrom.g6    23 graphs, Markström's file — identical up to isomorphism
data/near_miss_order26.g6    the graph described above
```

`filt.c` and `tutte_coxeter_lemma.py` are the two pieces most likely to be useful
elsewhere. Everything else is scaffolding.

## Sources

- Bondy & Vince, *Cycles in a graph whose lengths differ by one or two*, JGT 27 (1998)
- Markström, *Extremal graphs for some problems on cycles in graphs*, Congressus Numerantium 171 (2004) — and his data page, which carries the N ≤ 52 result
- Exoo, *Three Graphs and the Erdős-Gyárfás Conjecture*, [arXiv:1403.5636](https://arxiv.org/abs/1403.5636)
- Heckman & Krakovski, *Erdős-Gyárfás conjecture for cubic planar graphs*, EJC 20(2) (2013)
- Liu, Yu & Zhang, *Circumference of 3-connected cubic graphs*, [arXiv:1708.08865](https://arxiv.org/abs/1708.08865)
- Gao, Huo, Liu & Ma, [arXiv:1904.08126](https://arxiv.org/abs/1904.08126); Bucić, Gishboliner & Sudakov, [arXiv:2104.07633](https://arxiv.org/abs/2104.07633)
- Carr, [arXiv:2605.22844](https://arxiv.org/abs/2605.22844) and [arXiv:2508.19302](https://arxiv.org/abs/2508.19302)
- Narins, Pokrovskiy & Szabó, [arXiv:1408.5289](https://arxiv.org/abs/1408.5289)
- Brinkmann & Goedgebeur, `snarkhunter`; McKay & Piperno, `nauty`
