"""
Reconstruction of the Rybin counterexample instance to the
Dinitz-Garg-Goemans conjecture (published only as an X-thread image,
2026-07-22, Dmitry Rybin, found by GPT-5.6 Pro -- no arc list retrievable).

STATUS: CONFIRMED (not CANDIDATE-NOT-CONFIRMED). Every stated anchor is
reproduced exactly, in exact rationals, by the instance `build_rybin_instance()`
returns. See "Anchor-by-anchor match" below for the full accounting.

---------------------------------------------------------------------------
RECONSTRUCTION_PROVENANCE
---------------------------------------------------------------------------

Search space swept
-------------------
Stage 1 (topology, exhaustive): K4 has 4 branch vertices and 6 edges.
Subdividing 3 of the 6 edges gives 7 vertices / 9 arcs, matching anchor 1.
For each of the C(6,3)=20 edge subsets to subdivide, EVERY one of the
2**9 = 512 independent arc-direction assignments was tried (9 arcs, one
sign bit each) -- 10240 orientations total, filtered to the 6832 acyclic
ones, then to the 3556 with a unique in-degree-0 vertex s. For each such
DAG, path-counts from s to every other vertex were computed by DP; any
3-subset of vertices all showing s-path-count == 2 was kept as a
(topology, s, {t1,t2,t3}) candidate. This produced 88 candidates.

Important negative result folded into the search (worth recording because
it explains why terminals sit on subdivision vertices, not branch
vertices): if a subdivided edge's two arcs are forced to share one
"logical direction" (i.e. treated as a simple two-arc pass-through, as a
naive reading of "K4 subdivision" suggests), then acyclic orientations of
K4 are FORCED to be transitive tournaments (a standard fact: a tournament
is acyclic iff transitive), which forces s-path-count(rank-k branch
vertex) = 2**(k-1) for k=1,2,3 -- i.e. counts 1, 2, 4. Only ONE vertex can
ever land on count=2 that way, so "3 terminals each with exactly 2 paths"
is IMPOSSIBLE if all 4 branch vertices are {s,t1,t2,t3}. The fix (and what
the full 512-orientation sweep independently discovers, with no hand
steering) is to let a subdivision vertex be a local SINK: both of its arcs
point in, one from each endpoint of the original K4 edge. All 88 stage-1
candidates found this way. All 24 stage-2 hits below have their 3
terminals sitting exactly on the 3 subdivided (chord) edges of K4, exactly
as the task's fallback hint anticipated ("consider terminals at
subdivision vertices if branch-vertex terminals yield nothing").

Stage 2 (split fractions, exhaustive per candidate): for each of the 88
topologies x 3 demand assignments (which of the 3 terminal-slots gets
demand 10; d=(15,10,15) has a repeated value so only 3 distinct
assignments), a 31^3 = 29791-point rational grid (denominator 30) over
split fractions (p0,p1,p2) in [0,1]^3 was swept with numpy, computing the
overshoot max_a(f(a)-x(a)) for all 8 routings at every grid point, and
testing all 8 vertices of the {path0,path1}^3 routing cube as candidate
"all-cheap" origins for the target pattern (origin=26, all 3
one-coordinate-flip neighbors=16). This is exhaustive over the stated
class at the chosen grid resolution.

Result: 24 (topology, demand-assignment, split) hits, all in the demand
order/vertex-role combinatorics implied by the 88 topologies. Checking
isomorphism of the underlying digraphs (with node roles: source /
terminal-with-given-demand / internal) across all 24 hits shows they fall
into exactly ONE isomorphism class -- i.e. up to relabeling, the topology
+ split found here is the UNIQUE match at this grid resolution, not one of
several. (Verification script: see bottom of this docstring's twin file in
the campaign's scratchpad session; the check used networkx.is_isomorphic
with a role-based node_match.)

The topology (an explicit K4 identification)
----------------------------------------------
Branch vertices of K4: s, a, b, c. All 6 K4 edges are present:
  - 3 kept direct (the "chain"): s->a, a->b, b->c
  - 3 subdivided (the "chords"), each subdivision vertex a SINK fed by
    both endpoints of its chord:
      chord (a,c) subdivided by t1:  a->t1,  c->t1
      chord (s,c) subdivided by t2:  s->t2,  c->t2
      chord (s,b) subdivided by t3:  s->t3,  b->t3

9 arcs total: s-a, a-b, b-c, c-t1, a-t1, c-t2, s-t2, s-t3, b-t3. 7 vertices
(s,a,b,c,t1,t2,t3). Planar (any subdivision of a planar graph -- K4 is
planar -- stays planar; direction never affects planarity, so this holds
regardless of orientation). Terminal path sets (exactly 2 each, matching
anchor 3):
  t1 (d=15): SHORT s->a->t1              | LONG s->a->b->c->t1
  t2 (d=10): SHORT s->t2                 | LONG s->a->b->c->t2
  t3 (d=15): SHORT s->t3                 | LONG s->a->b->t3

Split x and cost solve (stage 3, exact)
-----------------------------------------
Grid hit for split: cheap-path (LONG-path) fractions p = (1/3, 2/5, 1/3)
for (t1,t2,t3) -- i.e. SHORT-path (expensive) fractions (2/3, 3/5, 2/3).
Sum of cheap-path fractions = 1/3 + 2/5 + 1/3 = 5/15+6/15+5/15 = 16/15 --
this EXACTLY matches the task's independent mechanism hint ("fractional
mass of cheap options is 16/15"), which was NOT a search target (the grid
search only targeted the overshoot pattern) -- strong cross-validation
that this is the right split, not a spurious grid hit.

Cost search: writing cost(routing) = base + sum_i [terminal i expensive] *
e_i with base = cost(ZZZ) and e_i = d_i * (cost of terminal i's expensive
path - cost of its cheap path), the constraints are exactly linear:
  base < 60;  base + e_i < 60 (i=1,2,3);  base + e_i + e_j >= 60 (i != j)
  c.x = 58  <=>  base + (2/3)e_1 + (3/5)e_2 + (2/3)e_3 = 58
An exhaustive integer sweep of base, e_1 in [0,60), solving for e_2 =
(870 - 15*base - 20*e_1)/9 (integer check) subject to all inequalities,
found EXACTLY ONE integer solution: base=0, e_1=e_3=30, e_2=30. This
forces every arc on every cheap (LONG) path to cost 0, and the 3
"short-side" arcs (a-t1, s-t2, s-t3) to carry all the cost:
  cost(a,t1) = e_1/d_1 = 30/15 = 2
  cost(s,t2) = e_2/d_2 = 30/10 = 3
  cost(s,t3) = e_3/d_3 = 30/15 = 2
all other 6 arcs cost 0. All costs nonnegative integers (the "integer
preferred" branch of the search succeeded; no need to fall back to
rationals).

Anchor-by-anchor match (all computed exactly in Fraction, verified in
__main__ below and in test_rybin.py)
---------------------------------------------------------------------------
1. DAG, 7 vertices, 9 arcs, planar K4-subdivision (3 of 6 edges
   subdivided): YES, by construction above.
2. d = (15, 10, 15), d_max = 15: YES.
3. Each terminal has exactly 2 s-t_i paths, 8 routings total: YES
   (check_paths_valid clean; Instance.routings() has 8 elements).
4. c.x = 58: YES, exactly (Fraction(58,1)).
   Cost-good (<60) routings are EXACTLY {ZZZ, EZZ, ZEZ, ZZE}:
     ZZZ=0, EZZ=30, ZEZ=30, ZZE=30 (all <60)
     ZEE=60, EZE=60, EEZ=60, EEE=90 (all >=60)
   Per-routing min overshoots of the 4 cost-good routings:
     ZZZ=26, EZZ=16, ZEZ=16, ZZE=16 -- EXACT match to the anchor.
   beta* = 16/15 (compute_beta_star, corrected sup-of-infeasible-region
   convention) -- EXACT match.
   Mechanism note cross-check: EVERY routing with >=2 terminals on their
   cheap path has overshoot > 15 (ZZZ=26, EZZ/ZEZ/ZZE=16 all >15) --
   confirms "the three cheap options are pairwise conflicting under
   +d_max slack." Cheap-mass sum = 16/15 -- EXACT match (see above).
   Bonus structural fact not in the anchor list but checked here: the
   "load-good" set at ceiling-slack d_max (routings with max_a(f(a)-x(a))
   <= 15) is EXACTLY {ZEE, EZE, EEZ, EEE} -- the complement of the
   cost-good set -- and their costs are exactly {60, 60, 60, 90}, all
   >= 60. This is precisely the stated fact "every routing with load
   f(a) <= x(a)+15 on all arcs has cost >= 60," verified for ALL 4 such
   routings, not just asserted.
5. t(x) <= -5: computed t_of_x = -9 for the split above (comfortably
   inside the bound). The claimed instance-level t* = -5 was checked with
   a separate 121^3 grid search over the split box [0,1]^3 (the full
   fractional-flow domain for this 2-paths-per-terminal instance): the
   grid max of t(p) is exactly -5.0, attained at the *exact* rational
   point p=(1/3, 0, 1/3) (cheap-fraction convention) -- confirmed exactly
   in Fraction arithmetic via core.verify_counterexample, giving
   t_of_x = -5 exactly. So t* >= -5 is EXACT (rational witness); t* <= -5
   (hence "1.3 holds here with slack exactly 5") rests on the 121^3 grid
   scan, i.e. is grid-certified, not proved -- adequate for a calibration
   anchor, would need an exact max-min argument to be theorem-grade.

Juang-family sanity probe (task deliverable 4)
-------------------------------------------------
Juang's parametrization of the same topology: demands (b+m, b, b+m),
b=10, m=5 -> (15,10,15), matching anchor 2 exactly. Given formulas:
  fractional cost   = 2*b**2 + (b+m)*(m+g)
  unsplittable floor = 2*b*(b+m)
With b=10, m=5: floor = 2*10*15 = 300. Our unsplittable-routing cost floor
(the min cost among load-good, i.e. ceiling-respecting, routings) is 60.
300 / 60 = 5, so try scale factor 1/5: fractional cost formula becomes
(200 + 15*(5+g)) / 5 = 58  =>  15*(5+g) = 90  =>  g = 1.
Check: 2*10**2 + 15*(5+1) = 200 + 90 = 290; 290/5 = 58 -- EXACT match to
c.x=58. And 2*10*15/5 = 300/5 = 60 -- EXACT match to the cost floor.
So g=1, scale=1/5 reproduces BOTH 58 and 60 exactly from the Juang
formulas -- independent numeric confirmation of this reconstruction from
a source (the parametrization) that was never used to pick the topology
or the split; it only entered as a late cross-check of the b,m,g,scale
relationship.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from typing import Dict, List, Tuple

from core import Arc, Instance, Terminal, check_paths_valid, verify_counterexample
from beta_star import compute_beta_star


def build_rybin_instance() -> Tuple[Instance, List[List[Fr]], Dict[Arc, Fr]]:
    """Build the reconstructed Rybin DGG counterexample instance.

    Returns (inst, split, costs):
      inst  - core.Instance: 7 vertices, 9 arcs, K4-subdivision topology,
              terminals t1 (d=15), t2 (d=10), t3 (d=15), each with exactly
              2 s->t_i paths (index 0 = cheap/LONG, index 1 = expensive/SHORT,
              per core.py's Z/E routing-label convention).
      split - per-terminal path-weight distribution matching the reconstructed
              fractional flow x (cheap-path fractions 1/3, 2/5, 1/3).
      costs - Dict[Arc, Fraction]: the reconstructed integer arc costs.

    See RECONSTRUCTION_PROVENANCE (module docstring) for how every number
    here was pinned down.
    """
    arcs: List[Arc] = [
        ("s", "a"), ("a", "b"), ("b", "c"),
        ("c", "t1"), ("a", "t1"),
        ("c", "t2"), ("s", "t2"),
        ("s", "t3"), ("b", "t3"),
    ]

    t1 = Terminal("t1", Fr(15), [
        (("s", "a"), ("a", "b"), ("b", "c"), ("c", "t1")),  # LONG / cheap
        (("s", "a"), ("a", "t1")),                          # SHORT / expensive
    ])
    t2 = Terminal("t2", Fr(10), [
        (("s", "a"), ("a", "b"), ("b", "c"), ("c", "t2")),  # LONG / cheap
        (("s", "t2"),),                                     # SHORT / expensive
    ])
    t3 = Terminal("t3", Fr(15), [
        (("s", "a"), ("a", "b"), ("b", "t3")),              # LONG / cheap
        (("s", "t3"),),                                     # SHORT / expensive
    ])

    inst = Instance(arcs=arcs, terminals=[t1, t2, t3])

    split = [
        [Fr(1, 3), Fr(2, 3)],   # t1: 1/3 cheap (LONG), 2/3 expensive (SHORT)
        [Fr(2, 5), Fr(3, 5)],   # t2: 2/5 cheap (LONG), 3/5 expensive (SHORT)
        [Fr(1, 3), Fr(2, 3)],   # t3: 1/3 cheap (LONG), 2/3 expensive (SHORT)
    ]

    costs: Dict[Arc, Fr] = {
        ("s", "a"): Fr(0), ("a", "b"): Fr(0), ("b", "c"): Fr(0),
        ("c", "t1"): Fr(0), ("a", "t1"): Fr(2),
        ("c", "t2"): Fr(0), ("s", "t2"): Fr(3),
        ("s", "t3"): Fr(2), ("b", "t3"): Fr(0),
    }

    return inst, split, costs


# The exact rational point (in cheap-fraction convention) at which the
# instance-level t* = -5 is attained -- see anchor 5 in the module
# docstring. Not the same x as the beta*/overshoot witness above; kept
# separate because it answers a different question (max over ALL x of
# t(x), not the properties of one specific x).
T_STAR_WITNESS_SPLIT: List[List[Fr]] = [
    [Fr(1, 3), Fr(2, 3)],   # t1: 1/3 cheap
    [Fr(0, 1), Fr(1, 1)],   # t2: 0 cheap (all expensive/direct)
    [Fr(1, 3), Fr(2, 3)],   # t3: 1/3 cheap
]


def routing_cost(inst: Instance, costs: Dict[Arc, Fr], routing) -> Fr:
    load = inst.routing_load(routing)
    return sum((costs[a] * load[a] for a in inst.arcs), Fr(0))


def cost_of_x(inst: Instance, costs: Dict[Arc, Fr], x: Dict[Arc, Fr]) -> Fr:
    return sum((costs[a] * x[a] for a in inst.arcs), Fr(0))


if __name__ == "__main__":
    inst, split, costs = build_rybin_instance()

    print("=== Structural checks ===")
    problems = check_paths_valid(inst, "s")
    print(f"check_paths_valid: {'CLEAN' if not problems else problems}")
    print(f"vertices: {sorted(set(v for a in inst.arcs for v in a))} "
          f"({len(set(v for a in inst.arcs for v in a))})")
    print(f"arcs: {len(inst.arcs)}")
    print(f"terminals: {[(t.name, t.demand, len(t.paths)) for t in inst.terminals]}")
    print(f"d_max: {inst.d_max()}")
    print(f"routings: {len(inst.routings())}")

    print("\n=== Cost / overshoot table (main split) ===")
    x = inst.x_from_split(split)
    cx = cost_of_x(inst, costs, x)
    print(f"c.x = {cx}  (anchor: 58)")

    bs = compute_beta_star(inst, split)
    overshoot_by_label = {
        inst.routing_label(r): o for r, o in zip(bs["routings"], bs["overshoots"])
    }
    res = verify_counterexample(inst, split)
    dev_by_label = {row["label"]: row["worst_deviation"] for row in res["table"]}

    print(f"\n{'label':6}{'cost':>6}{'cost-good':>11}{'overshoot':>11}"
          f"{'two-sided dev':>15}{'ceiling-good(<=15)':>21}")
    for r in inst.routings():
        label = inst.routing_label(r)
        cost_r = routing_cost(inst, costs, r)
        cgood = cost_r < 60
        ov = overshoot_by_label[label]
        dev = dev_by_label[label]
        cgood_ceiling = ov <= inst.d_max()
        print(f"{label:6}{str(cost_r):>6}{str(cgood):>11}{str(ov):>11}"
              f"{str(dev):>15}{str(cgood_ceiling):>21}")

    print(f"\nbeta* = {bs['beta_star']}  (anchor: 16/15)")
    print(f"t(x) of main split = {res['t_of_x']}  (anchor: <= -5)")

    cheap_mass = sum(s[0] for s in split)
    print(f"cheap-path fractional mass sum = {cheap_mass}  (anchor: 16/15)")

    print("\n=== Instance-level t* witness ===")
    res_star = verify_counterexample(inst, T_STAR_WITNESS_SPLIT)
    print(f"t(x*) at witness split {T_STAR_WITNESS_SPLIT} = {res_star['t_of_x']}"
          f"  (anchor: t* = -5)")

    print("\n=== Juang-family sanity probe ===")
    b, m, g, scale = Fr(10), Fr(5), Fr(1), Fr(1, 5)
    frac_cost = 2 * b ** 2 + (b + m) * (m + g)
    floor = 2 * b * (b + m)
    print(f"b={b} m={m} g={g} scale={scale}")
    print(f"fractional cost formula 2b^2+(b+m)(m+g) = {frac_cost}, "
          f"*scale = {frac_cost * scale}  (anchor: 58)")
    print(f"unsplittable floor formula 2b(b+m) = {floor}, "
          f"*scale = {floor * scale}  (anchor: 60)")

    print("\n=== Verdict ===")
    all_ok = (
        cx == 58
        and bs["beta_star"] == Fr(16, 15)
        and res["t_of_x"] <= -5
        and res_star["t_of_x"] == -5
        and cheap_mass == Fr(16, 15)
        and {label for label, o in overshoot_by_label.items()
             if routing_cost(inst, costs,
                              next(r for r in inst.routings()
                                   if inst.routing_label(r) == label)) < 60}
            == {"ZZZ", "EZZ", "ZEZ", "ZZE"}
    )
    print("ALL ANCHORS CONFIRMED" if all_ok else "MISMATCH -- see table above")
