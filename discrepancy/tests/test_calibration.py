"""Calibration gates for the discrepancy engines (issue #166).

Every engine must reproduce known ground truth before any record claim:

Komlós side
  G1  identity matrix -> disc = 1 exactly (any n)
  G2  Kunisky k=1 (n=2) -> disc = sqrt(2) exactly
  G3  exact_disc_sqrt2 == brute force (pure-python Fraction-free) on the
      Kunisky matrices k <= 3
  G4  exact_disc_fraction == float_disc on random rational matrices
  G5  delta(A-hat) lower bound: exact disc >= delta for k <= 3 (Prop 2.3)

Beck-Fiala side
  G6  triangle system (t=2) -> disc = 2; single set -> disc = |S| mod 2 ...
      concretely {1,2,3} alone -> 1
  G7  sat_disc_geq agrees with disc_exact on random systems
  G8  Fano plane (t=3): disc computed identically by SAT and enumeration
  G9  CEGAR reproduces D(2, n<=5) = 2 (exists k=2, not exists k=3)
"""

import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from komlos import (
    all_sign_vectors,
    exact_disc_fraction,
    exact_disc_sqrt2,
    float_disc,
    kunisky_delta_lower_bound,
    kunisky_normalized_int_pair,
    kunisky_tree_matrix,
    sign_p_q_sqrt2,
    cmp_abs_sqrt2,
    sqrt2_to_float,
)
from beck_fiala import (
    cegar_exists,
    disc_exact,
    max_degree,
    sat_disc_geq,
)

SQRT2 = 2 ** 0.5


def test_sign_arith():
    assert sign_p_q_sqrt2(0, 0) == 0
    assert sign_p_q_sqrt2(3, -2) == 1     # 3 > 2*sqrt2 ~ 2.83
    assert sign_p_q_sqrt2(2, -2) == -1    # 2 < 2*sqrt2
    assert sign_p_q_sqrt2(-3, 2) == -1
    assert sign_p_q_sqrt2(-2, 2) == 1
    assert cmp_abs_sqrt2((1, 1), (2, 0)) > 0     # 1+sqrt2 > 2
    assert cmp_abs_sqrt2((-1, -1), (2, 0)) > 0   # abs value
    assert cmp_abs_sqrt2((0, 1), (1, 0)) > 0     # sqrt2 > 1
    assert cmp_abs_sqrt2((2, -1), (2, -1)) == 0
    print("PASS sign/compare arithmetic")


def test_g1_identity():
    n = 6
    A = np.eye(n, dtype=np.int64)
    B = np.zeros((n, n), dtype=np.int64)
    r = exact_disc_sqrt2(A, B, 1)
    assert r["disc_exact_pair"] == (1, 0) and r["scale"] == 1
    print("PASS G1 identity disc = 1")


def test_g2_kunisky_n2():
    A, B, scale = kunisky_normalized_int_pair(1)
    r = exact_disc_sqrt2(A, B, scale)
    # disc = sqrt2: pair (u, w) with (u + w*sqrt2)/2 = sqrt2 -> (0, 2)/2
    u, w = r["disc_exact_pair"]
    assert abs(sqrt2_to_float(u, w) / scale - SQRT2) < 1e-12
    # exact: value^2 must equal 2: ((u+w*sqrt2)/s)^2 == 2
    p, q = u * u + 2 * w * w, 2 * u * w
    assert p == 2 * scale * scale and q == 0
    print("PASS G2 Kunisky n=2 disc = sqrt(2) exactly")


def brute_disc_float(A, B, scale):
    n = A.shape[1]
    best = None
    for bits in range(1 << (n - 1)):
        eps = [1] + [1 - 2 * ((bits >> i) & 1) for i in range(n - 1)]
        mx = 0.0
        for i in range(A.shape[0]):
            u = sum(int(A[i, j]) * eps[j] for j in range(n))
            w = sum(int(B[i, j]) * eps[j] for j in range(n))
            mx = max(mx, abs(u + w * SQRT2))
        v = mx / scale
        if best is None or v < best:
            best = v
    return best


def test_g3_bruteforce_agreement():
    for k in (1, 2, 3):
        A, B, scale = kunisky_normalized_int_pair(k)
        r = exact_disc_sqrt2(A, B, scale)
        bf = brute_disc_float(A, B, scale)
        assert abs(r["disc_float"] - bf) < 1e-9, (k, r["disc_float"], bf)
    print("PASS G3 exact engine == brute force, Kunisky k=1..3")


def test_g4_fraction_engine():
    rng = np.random.default_rng(7)
    for trial in range(5):
        d, n = 4, 5
        V = rng.integers(-4, 5, size=(d, n))
        den = int(rng.integers(3, 9))
        VF = [[Fraction(int(V[i, j]), den) for j in range(n)]
              for i in range(d)]
        r = exact_disc_fraction(VF)
        f = float_disc(V.astype(np.float64) / den)
        assert abs(r["disc_float"] - f) < 1e-9
    print("PASS G4 Fraction engine == float engine, random matrices")


def test_g5_delta_bound():
    for k in (1, 2, 3):
        A, B, scale = kunisky_normalized_int_pair(k)
        r = exact_disc_sqrt2(A, B, scale)
        delta = kunisky_delta_lower_bound(k)
        assert r["disc_float"] >= delta - 1e-9, (k, r["disc_float"], delta)
    print("PASS G5 exact disc >= delta lower bound (Prop 2.3)")


def test_g6_small_systems():
    # triangle: elements = 3 edges, sets = vertices (pairs of edges), t = 2
    tri = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert max_degree(tri) == 2
    d, _ = disc_exact(tri)
    assert d == 2, d
    single = np.array([[1, 1, 1]])
    d, _ = disc_exact(single)
    assert d == 1
    print("PASS G6 triangle disc = 2, single 3-set disc = 1")


def test_g7_sat_vs_enumeration():
    rng = np.random.default_rng(11)
    for trial in range(8):
        m, n = int(rng.integers(2, 7)), int(rng.integers(3, 9))
        M = (rng.random((m, n)) < 0.45).astype(np.int64)
        d, _ = disc_exact(M)
        assert sat_disc_geq(M, d) is True or d == 0
        assert sat_disc_geq(M, d + 1) is False
    print("PASS G7 SAT certificate == enumeration on random systems")


def test_g8_fano():
    lines = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5), (1, 4, 6),
             (2, 3, 6), (2, 4, 5)]
    M = np.zeros((7, 7), dtype=np.int64)
    for i, ln in enumerate(lines):
        M[i, list(ln)] = 1
    assert max_degree(M) == 3
    d, _ = disc_exact(M)
    assert sat_disc_geq(M, d) and not sat_disc_geq(M, d + 1)
    print(f"PASS G8 Fano plane disc = {d}, SAT and enumeration agree")


def test_g9_cegar_t2():
    for n in (3, 4, 5):
        r2 = cegar_exists(2, n, 2)
        assert r2["exists"], f"D(2,{n}) >= 2 should exist"
        wd, _ = disc_exact(r2["witness"])
        assert wd >= 2 and max_degree(r2["witness"]) <= 2
        r3 = cegar_exists(2, n, 3)
        assert not r3["exists"], f"D(2,{n}) >= 3 should NOT exist"
    print("PASS G9 CEGAR: D(2, n<=5) = 2 exactly")


if __name__ == "__main__":
    test_sign_arith()
    test_g1_identity()
    test_g2_kunisky_n2()
    test_g3_bruteforce_agreement()
    test_g4_fraction_engine()
    test_g5_delta_bound()
    test_g6_small_systems()
    test_g7_sat_vs_enumeration()
    test_g8_fano()
    test_g9_cegar_t2()
    print("\nALL CALIBRATION GATES GREEN")
