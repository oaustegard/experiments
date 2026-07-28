"""
Assertions for every anchor claimed about the reconstructed Rybin DGG
counterexample instance. Run: python3 test_rybin.py (from this directory).

Anchors under test (see rybin.py's RECONSTRUCTION_PROVENANCE docstring for
the full derivation of each):
  1. DAG, 7 vertices, 9 arcs, planar K4-subdivision (3 of 6 edges subdivided)
  2. d = (15, 10, 15), d_max = 15
  3. each terminal has exactly 2 s->t_i paths; 8 routings total
  4. c.x = 58; cost-good (<60) set is EXACTLY {ZZZ, EZZ, ZEZ, ZZE};
     per-routing min overshoots of those four are (26, 16, 16, 16);
     beta* = 16/15; cheap-path fractional mass = 16/15; every routing with
     load <= x+15 on all arcs has cost >= 60
  5. t(x) <= -5 for the main split; instance-level t* = -5, attained
     exactly at an explicit witness split
  6. Juang-family formulas (b=10, m=5, g=1, scale=1/5) reproduce 58 and 60
No floats anywhere in the certificate path -- everything is fractions.Fraction.
"""
from fractions import Fraction as Fr

from core import check_paths_valid, verify_counterexample
from beta_star import compute_beta_star
from rybin import (
    T_STAR_WITNESS_SPLIT,
    build_rybin_instance,
    cost_of_x,
    routing_cost,
)


def test_no_floats_in_module():
    import inspect
    import rybin
    src = inspect.getsource(rybin)
    # 'float(' calls appear only in beta_star's verbose diagnostic path,
    # never in rybin.py itself.
    assert "float(" not in src, "rybin.py must not touch floats in the certificate path"


def test_structure():
    inst, split, costs = build_rybin_instance()
    vertices = set(v for a in inst.arcs for v in a)
    assert len(vertices) == 7, vertices
    assert len(inst.arcs) == 9, inst.arcs
    assert len(inst.terminals) == 3
    assert check_paths_valid(inst, "s") == []

    # K4 subdivision: 3 direct branch-branch arcs (the chain) + 3 chords,
    # each chord subdivided by a terminal fed from BOTH chord endpoints.
    branch = {"s", "a", "b", "c"}
    direct_branch_arcs = [a for a in inst.arcs if a[0] in branch and a[1] in branch]
    assert set(direct_branch_arcs) == {("s", "a"), ("a", "b"), ("b", "c")}, direct_branch_arcs
    # each terminal has in-degree exactly 2, both from branch vertices
    for term in ("t1", "t2", "t3"):
        into = [a for a in inst.arcs if a[1] == term]
        assert len(into) == 2, (term, into)
        assert all(a[0] in branch for a in into), (term, into)


def test_demands():
    inst, split, costs = build_rybin_instance()
    demands = [t.demand for t in inst.terminals]
    assert demands == [Fr(15), Fr(10), Fr(15)], demands
    assert inst.d_max() == Fr(15)


def test_exactly_two_paths_eight_routings():
    inst, split, costs = build_rybin_instance()
    for t in inst.terminals:
        assert len(t.paths) == 2, (t.name, t.paths)
    assert len(inst.routings()) == 8


def test_cost_of_x_is_58():
    inst, split, costs = build_rybin_instance()
    x = inst.x_from_split(split)
    assert cost_of_x(inst, costs, x) == Fr(58)


def test_cost_good_set_is_exactly_ZZZ_EZZ_ZEZ_ZZE():
    inst, split, costs = build_rybin_instance()
    good = set()
    bad = set()
    for r in inst.routings():
        label = inst.routing_label(r)
        c = routing_cost(inst, costs, r)
        (good if c < 60 else bad).add(label)
    assert good == {"ZZZ", "EZZ", "ZEZ", "ZZE"}, good
    assert bad == {"ZEE", "EZE", "EEZ", "EEE"}, bad


def test_overshoot_pattern_26_16_16_16():
    inst, split, costs = build_rybin_instance()
    bs = compute_beta_star(inst, split)
    overshoot = {inst.routing_label(r): o for r, o in zip(bs["routings"], bs["overshoots"])}
    assert overshoot["ZZZ"] == 26, overshoot
    assert overshoot["EZZ"] == 16, overshoot
    assert overshoot["ZEZ"] == 16, overshoot
    assert overshoot["ZZE"] == 16, overshoot


def test_beta_star_is_16_over_15():
    inst, split, costs = build_rybin_instance()
    bs = compute_beta_star(inst, split)
    assert bs["beta_star"] == Fr(16, 15), bs["beta_star"]


def test_cheap_mass_is_16_over_15():
    inst, split, costs = build_rybin_instance()
    cheap_mass = sum(s[0] for s in split)
    assert cheap_mass == Fr(16, 15), cheap_mass


def test_every_ceiling_good_routing_costs_at_least_60():
    """'EVERY routing with load f(a) <= x(a) + d_max on all arcs has cost
    >= 60' -- verified directly, not merely asserted."""
    inst, split, costs = build_rybin_instance()
    x = inst.x_from_split(split)
    dmax = inst.d_max()
    checked_any = False
    for r in inst.routings():
        load = inst.routing_load(r)
        ceiling_good = all(load[a] <= x[a] + dmax for a in inst.arcs)
        if ceiling_good:
            checked_any = True
            c = routing_cost(inst, costs, r)
            assert c >= 60, (inst.routing_label(r), c)
    assert checked_any, "no ceiling-good routing found -- test is vacuous, investigate"
    # and per the reconstruction, the ceiling-good set is exactly the
    # complement of the cost-good set:
    ceiling_good_labels = {
        inst.routing_label(r) for r in inst.routings()
        if all(inst.routing_load(r)[a] <= x[a] + dmax for a in inst.arcs)
    }
    assert ceiling_good_labels == {"ZEE", "EZE", "EEZ", "EEE"}, ceiling_good_labels


def test_t_of_main_split_at_most_minus_5():
    inst, split, costs = build_rybin_instance()
    res = verify_counterexample(inst, split)
    assert res["t_of_x"] <= -5, res["t_of_x"]


def test_t_star_witness_hits_minus_5_exactly():
    inst, split, costs = build_rybin_instance()
    res = verify_counterexample(inst, T_STAR_WITNESS_SPLIT)
    assert res["t_of_x"] == -5, res["t_of_x"]


def test_juang_family_sanity_probe():
    b, m, g, scale = Fr(10), Fr(5), Fr(1), Fr(1, 5)
    frac_cost = 2 * b ** 2 + (b + m) * (m + g)
    floor = 2 * b * (b + m)
    assert frac_cost * scale == Fr(58), frac_cost * scale
    assert floor * scale == Fr(60), floor * scale


TESTS = [
    test_no_floats_in_module,
    test_structure,
    test_demands,
    test_exactly_two_paths_eight_routings,
    test_cost_of_x_is_58,
    test_cost_good_set_is_exactly_ZZZ_EZZ_ZEZ_ZZE,
    test_overshoot_pattern_26_16_16_16,
    test_beta_star_is_16_over_15,
    test_cheap_mass_is_16_over_15,
    test_every_ceiling_good_routing_costs_at_least_60,
    test_t_of_main_split_at_most_minus_5,
    test_t_star_witness_hits_minus_5_exactly,
    test_juang_family_sanity_probe,
]


if __name__ == "__main__":
    failures = []
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print(f"FAIL {fn.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} tests FAILED")
        raise SystemExit(1)
    print(f"all {len(TESTS)} tests passed")
