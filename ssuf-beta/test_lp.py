"""Sanity tests for the exact rational LP feasibility solver, independent of SSUF."""

from fractions import Fraction as Fr
from engine import exact_lp_feasible


def check(name, ok, expected):
    status = "PASS" if ok == expected else "FAIL"
    print(f"[{status}] {name} (feasible={ok}, expected={expected})")
    assert ok == expected, name


def test_1d_inside():
    # single "arc" constraint: convex combo of loads {0, 2} should hit x=1.
    # A_rows encodes: w . [0,2] <= 1  and  -w.[0,2] <= -1  (equality)
    A = [[Fr(0), Fr(2)], [Fr(0), Fr(-2)]]
    b = [Fr(1), Fr(-1)]
    ok, w = exact_lp_feasible(A, b, 2)
    check("1d inside hull", ok, True)
    if ok:
        assert w[0] * 0 + w[1] * 2 == 1
        assert sum(w) == 1
        assert all(wi >= 0 for wi in w)


def test_1d_outside():
    # loads {3, 5}, target x=1 -- outside [3,5], must be infeasible.
    A = [[Fr(3), Fr(5)], [Fr(-3), Fr(-5)]]
    b = [Fr(1), Fr(-1)]
    ok, w = exact_lp_feasible(A, b, 2)
    check("1d outside hull", ok, False)


def test_2d_triangle_centroid():
    # 3 routings at "loads" on two arcs: (0,0), (3,0), (0,3). Target = centroid (1,1).
    # centroid = (1/3)(0,0)+(1/3)(3,0)+(1/3)(0,3) = (1,1) -- must be feasible.
    routing_loads = [(Fr(0), Fr(0)), (Fr(3), Fr(0)), (Fr(0), Fr(3))]
    target = (Fr(1), Fr(1))
    A = []
    b = []
    for arc_idx in range(2):
        A.append([routing_loads[i][arc_idx] for i in range(3)])
        b.append(target[arc_idx])
        A.append([-routing_loads[i][arc_idx] for i in range(3)])
        b.append(-target[arc_idx])
    ok, w = exact_lp_feasible(A, b, 3)
    check("2d triangle centroid", ok, True)


def test_2d_triangle_outside():
    # same triangle, target (2,2) -- outside the triangle (x+y<=3 boundary for
    # points with x,y>=0 is satisfied at (2,2) since 2+2=4>3... wait check hull:
    # hull is {(x,y): x>=0,y>=0,x+y<=3}. (2,2) has x+y=4>3 -> outside.
    routing_loads = [(Fr(0), Fr(0)), (Fr(3), Fr(0)), (Fr(0), Fr(3))]
    target = (Fr(2), Fr(2))
    A = []
    b = []
    for arc_idx in range(2):
        A.append([routing_loads[i][arc_idx] for i in range(3)])
        b.append(target[arc_idx])
        A.append([-routing_loads[i][arc_idx] for i in range(3)])
        b.append(-target[arc_idx])
    ok, w = exact_lp_feasible(A, b, 3)
    check("2d triangle outside (must be infeasible)", ok, False)


def test_odd_triangle_stable_set_gap():
    # THE mechanism behind beta*=16/15: 3 pairwise-conflicting zero-cost options
    # with fractional mass 16/15 > 1 means: no single option alone reaches the
    # target, but a fractional (non-integral) combination overshoots weight 1.
    # Minimal reproduction: 3 "arcs" (one per pairwise conflict edge of a triangle
    # conflict graph), 3 routings, routing i loads arc i and arc (i+1)%3 to 1 each
    # (so it "occupies" both edges of the triangle it touches), target x = (2/3,2/3,2/3)
    # -- the fractional stable-set-relaxation optimum of a triangle (independence
    # number 1 integrally, 3/2 fractionally httpsis actually the vertex cover /
    # independent set LP relaxation gap). This is a structural smoke test of the
    # solver on a genuine "small fractional mass, no single vertex covers it"
    # instance, not a transcription of the Rybin certificate.
    routing_loads = [
        (Fr(1), Fr(1), Fr(0)),
        (Fr(0), Fr(1), Fr(1)),
        (Fr(1), Fr(0), Fr(1)),
    ]
    target = (Fr(2, 3), Fr(2, 3), Fr(2, 3))
    A = []
    b = []
    for arc_idx in range(3):
        A.append([routing_loads[i][arc_idx] for i in range(3)])
        b.append(target[arc_idx])
        A.append([-routing_loads[i][arc_idx] for i in range(3)])
        b.append(-target[arc_idx])
    ok, w = exact_lp_feasible(A, b, 3)
    check("symmetric triangle centroid (equal weights 1/3 each)", ok, True)


if __name__ == "__main__":
    test_1d_inside()
    test_1d_outside()
    test_2d_triangle_centroid()
    test_2d_triangle_outside()
    test_odd_triangle_stable_set_gap()
    print("\nALL LP SOLVER SANITY TESTS PASSED")
