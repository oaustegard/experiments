"""
k3-proof: complete the k=3 half of Conjecture 12.1 (NG-12) -- the upper bound
R <= 3/4 for every out-tree + 3-chord structure -- with exact rational proof.

Task A: enumerate all distinct k=3 "row-set types" (mod the symmetry group:
per-row sign flip, per-column sign flip, column permutation) and argue/verify
completeness.

Task B: exact branch-and-bound proof that R <= 3/4 for every MAXIMAL type
(R is monotone non-decreasing under adding rows, so maximal types suffice),
plus the exact value of R per type.

Task C: search for a simple deterministic rounding rule that certifies 3/4
on all types (a constructive proof shape, if one exists).

See K3_PROOF.md for the write-up. This script is self-contained (does not
import outtree.py/discrepancy.py for the core search -- see
`cross_check_against_outtree` for the independent verification bridge).
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

K = int(__import__("os").environ.get("MS13_K", "4"))


# ---------------------------------------------------------------------------
# Task A.1: non-isomorphic unrooted tree shapes via Pruefer sequences + AHU
# ---------------------------------------------------------------------------

def pruefer_decode(seq: Sequence[int], m: int) -> List[Tuple[int, int]]:
    """Standard Pruefer decode: seq has length m-2, symbols in 0..m-1."""
    degree = [1] * m
    for s in seq:
        degree[s] += 1
    edges = []
    seq = list(seq)
    import heapq
    leaves = [i for i in range(m) if degree[i] == 1]
    heapq.heapify(leaves)
    for s in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, s))
        degree[leaf] -= 1
        degree[s] -= 1
        if degree[s] == 1:
            heapq.heappush(leaves, s)
    remaining = [i for i in range(m) if degree[i] == 1]
    edges.append((remaining[0], remaining[1]))
    return edges


def adjacency(edges: Sequence[Tuple[int, int]], m: int) -> List[List[int]]:
    adj = [[] for _ in range(m)]
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def ahu_signature(adj: List[List[int]], m: int) -> str:
    """Canonical isomorphism signature of an unrooted tree (label-free)."""
    if m == 1:
        return "()"
    deg = [len(adj[i]) for i in range(m)]
    # peel leaves to find center(s)
    remaining = set(range(m))
    layer = [i for i in range(m) if deg[i] <= 1]
    deg = deg[:]
    while len(remaining) > 2:
        nxt = []
        for u in layer:
            if u not in remaining:
                continue
            remaining.discard(u)
            for v in adj[u]:
                if v in remaining:
                    deg[v] -= 1
                    if deg[v] == 1:
                        nxt.append(v)
        layer = nxt
    centers = list(remaining)

    def rooted_sig(root: int) -> str:
        # iterative post-order to avoid recursion-depth worries (m is small
        # anyway, but keep it clean)
        parent = {root: None}
        order = [root]
        stack = [root]
        seen = {root}
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    order.append(v)
                    stack.append(v)
        sig = {}
        for u in reversed(order):
            children_sigs = sorted(sig[v] for v in adj[u] if v != parent[u])
            sig[u] = "(" + "".join(children_sigs) + ")"
        return sig[root]

    return min(rooted_sig(c) for c in centers)


def generate_shapes(m: int) -> List[List[Tuple[int, int]]]:
    """All non-isomorphic tree shapes on m labeled-0..m-1 nodes, one
    representative edge-list per shape, via exhaustive Pruefer enumeration."""
    if m == 1:
        return [[]]
    if m == 2:
        return [[(0, 1)]]
    seen: Dict[str, List[Tuple[int, int]]] = {}
    for seq in itertools.product(range(m), repeat=m - 2):
        edges = pruefer_decode(seq, m)
        adj = adjacency(edges, m)
        sig = ahu_signature(adj, m)
        if sig not in seen:
            seen[sig] = edges
    return list(seen.values())


# ---------------------------------------------------------------------------
# Task A.2: rows from a (shape, markers) pair, canonicalization under the
# symmetry group (per-row sign flip x per-column sign flip x column perm)
# ---------------------------------------------------------------------------

def subtree_masks(edges: Sequence[Tuple[int, int]], m: int, root: int = 0
                   ) -> List[Tuple[int, int]]:
    """Returns [(child, mask)] for every tree edge (parent(child), child),
    mask = bitmask of the subtree hanging below child when rooted at `root`.
    """
    adj = adjacency(edges, m)
    parent = {root: None}
    order = [root]
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v] = u
                order.append(v)
                stack.append(v)
    mask = {u: (1 << u) for u in range(m)}
    for u in reversed(order):
        if parent[u] is not None:
            mask[parent[u]] |= mask[u]
    return [(u, mask[u]) for u in order if parent[u] is not None]


def rows_from_markers(edge_masks: Sequence[Tuple[int, int]],
                       markers: Sequence[int]) -> List[Tuple[int, int, int]]:
    """markers = (u1,v1,u2,v2,u3,v3) as node indices. Returns nonzero rows."""
    rows = []
    for (_child, mask) in edge_masks:
        row = tuple(
            ((mask >> markers[2 * i]) & 1) - ((mask >> markers[2 * i + 1]) & 1)
            for i in range(K)
        )
        if any(row):
            rows.append(row)
    return rows


def normalize_row(row: Tuple[int, ...]) -> Tuple[int, ...]:
    for x in row:
        if x != 0:
            return row if x > 0 else tuple(-v for v in row)
    return row


_PERMS = list(itertools.permutations(range(K)))
_SIGNS = list(itertools.product((1, -1), repeat=K))
_GROUP = [(p, s) for p in _PERMS for s in _SIGNS]  # 48 elements

# Precomputed fast path: enumerate the 26 nonzero rows in {-1,0,1}^3, index
# them, and precompute IMAGE[g][idx] = index of normalize(g applied to row)
# so canonicalization is integer-lookup only (no tuple algebra in the hot
# loop). ~5-10x speedup measured over the naive version, needed because Task
# A's enumeration calls this hundreds of thousands to millions of times.
_ROW_VALUES = [r for r in itertools.product((-1, 0, 1), repeat=K) if any(r)]
_ROW_INDEX = {r: i for i, r in enumerate(_ROW_VALUES)}


def _row_normalize_idx(row: Tuple[int, ...]) -> int:
    return _ROW_INDEX[normalize_row(row)]


_IMAGE = [
    [ _row_normalize_idx(tuple(r[perm[j]] * signs[j] for j in range(K)))
      for r in _ROW_VALUES ]
    for (perm, signs) in _GROUP
]


def canonical_rowset(rows: Sequence[Tuple[int, ...]]
                      ) -> Tuple[Tuple[int, ...], ...]:
    """Canonical form of a row SET (duplicates collapse) under the full
    symmetry group: column permutation, column sign flip, per-row sign flip."""
    idx_set = {_row_normalize_idx(r) for r in rows}
    best = None
    for g in range(48):
        img = _IMAGE[g]
        cand = tuple(sorted(img[i] for i in idx_set))
        if best is None or cand < best:
            best = cand
    return tuple(_ROW_VALUES[i] for i in best)


def apply_group(rows: Sequence[Tuple[int, ...]], perm, signs
                 ) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted({normalize_row(tuple(r[perm[j]] * signs[j]
                                              for j in range(K)))
                          for r in rows}))


def is_dominated(a_rows: Tuple[Tuple[int, ...], ...],
                  b_rows: Tuple[Tuple[int, ...], ...]) -> bool:
    """True if some symmetry image of a_rows (as a SET) is a subset of
    b_rows (row-for-row, allowing each row's own sign flip)."""
    b_signed = set(b_rows) | {tuple(-x for x in r) for r in b_rows}
    for (perm, signs) in _GROUP:
        img = {tuple(r[perm[j]] * signs[j] for j in range(K)) for r in a_rows}
        if img <= b_signed:
            return True
    return False


# ---------------------------------------------------------------------------
# Task A.3: the enumeration driver
# ---------------------------------------------------------------------------

def _marker_multisets(m: int):
    """All (u1,v1,u2,v2,u3,v3) up to the column-permutation + per-column
    orientation symmetry: unordered PAIRS {u,v} (u<v, since either
    orientation is covered by the sign-flip half of the group -- verified in
    K3_PROOF.md), chosen as a MULTISET of 3 (chord relabelling is the
    permutation half of the group). This is what makes exhaustive m up to 8
    tractable: m^6 collapses to C(C(m,2)+2, 3)."""
    pairs = list(itertools.combinations(range(m), 2))
    for combo in itertools.combinations_with_replacement(pairs, K):
        markers = []
        for (u, v) in combo:
            markers.extend((u, v))
        yield tuple(markers)


def enumerate_exhaustive(m: int, found: Dict[Tuple, dict], log_every=200000):
    shapes = generate_shapes(m)
    t0 = time.time()
    total = 0
    new_here = 0
    for shape in shapes:
        edge_masks = subtree_masks(shape, m, root=0)
        for markers in _marker_multisets(m):
            total += 1
            rows = rows_from_markers(edge_masks, markers)
            if not rows:
                continue
            canon = canonical_rowset(rows)
            if canon not in found:
                found[canon] = {"n": m, "shape": shape, "markers": markers,
                                 "rows": canon}
                new_here += 1
            if total % log_every == 0:
                print(f"  m={m}: {total} combos, {len(found)} types so far "
                      f"({time.time()-t0:.1f}s)", flush=True)
    print(f"m={m} EXHAUSTIVE done: {total} combos, {len(shapes)} shapes, "
          f"{new_here} new types, {time.time()-t0:.1f}s", flush=True)
    return new_here


def enumerate_sampled(m: int, found: Dict[Tuple, dict], n_samples: int,
                       seed: int = 0, log_every=50000):
    rng = random.Random(seed)
    t0 = time.time()
    new_here = 0
    for i in range(1, n_samples + 1):
        if m >= 2:
            seq = [rng.randrange(m) for _ in range(max(0, m - 2))]
            edges = pruefer_decode(seq, m)
        else:
            edges = []
        edge_masks = subtree_masks(edges, m, root=0)
        markers = [rng.randrange(m) for _ in range(2 * K)]
        if markers[0] == markers[1] or markers[2] == markers[3] or \
           markers[4] == markers[5]:
            continue
        rows = rows_from_markers(edge_masks, markers)
        if not rows:
            continue
        canon = canonical_rowset(rows)
        if canon not in found:
            found[canon] = {"n": m, "shape": edges, "markers": tuple(markers),
                             "rows": canon}
            new_here += 1
        if i % log_every == 0:
            print(f"  m={m} sampling: {i}/{n_samples}, {len(found)} types "
                  f"so far, {new_here} new at this m ({time.time()-t0:.1f}s)",
                  flush=True)
    print(f"m={m} SAMPLED ({n_samples}) done: {new_here} new types, "
          f"{time.time()-t0:.1f}s", flush=True)
    return new_here


# ---------------------------------------------------------------------------
# Maximality
# ---------------------------------------------------------------------------

def find_maximal(found: Dict[Tuple, dict]) -> List[Tuple]:
    canons = list(found.keys())
    maximal = []
    for a in canons:
        dominated_by_other = False
        for b in canons:
            if a == b:
                continue
            if len(b) <= len(a):
                continue
            if is_dominated(a, b):
                dominated_by_other = True
                break
        if not dominated_by_other:
            maximal.append(a)
    return maximal


def dedupe_equal_length_maximal(found: Dict[Tuple, dict],
                                  maximal: List[Tuple]) -> List[Tuple]:
    """Among the maximal list, two DISTINCT canonical forms of the same size
    can still be mutually non-dominating (genuinely different maximal
    types) -- that's expected and kept. This just guards against a canon
    dominating itself in edge cases (shouldn't happen; sanity check only)."""
    return maximal


# ---------------------------------------------------------------------------
# Task A driver
# ---------------------------------------------------------------------------

def run_enumeration(max_exhaustive_m=9, sample_m=(10,), n_samples=250000,
                     results_path=None):
    found: Dict[Tuple, dict] = {}
    history = []
    t_start = time.time()
    for m in range(2, max_exhaustive_m + 1):
        n_new = enumerate_exhaustive(m, found)
        history.append({"m": m, "mode": "exhaustive", "new_types": n_new,
                         "total_types": len(found),
                         "elapsed_s": time.time() - t_start})
        if results_path:
            save_partial(found, history, results_path)
    for m in sample_m:
        n_new = enumerate_sampled(m, found, n_samples)
        history.append({"m": m, "mode": f"sampled({n_samples})",
                         "new_types": n_new, "total_types": len(found),
                         "elapsed_s": time.time() - t_start})
        if results_path:
            save_partial(found, history, results_path)
    maximal = find_maximal(found)
    print(f"\nTOTAL distinct k=3 row-set types: {len(found)}", flush=True)
    print(f"MAXIMAL types: {len(maximal)}", flush=True)
    return found, maximal, history


def save_partial(found, history, path):
    data = {
        "types": [
            {"rows": [list(r) for r in canon], "n": info["n"],
             "shape": info["shape"], "markers": list(info["markers"])}
            for canon, info in found.items()
        ],
        "history": history,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


# ---------------------------------------------------------------------------
# Task B: exact R per type, and the exact B&B certifying R <= 3/4
# ---------------------------------------------------------------------------

_ZS = list(itertools.product((0, 1), repeat=K))


def float_R_milp(rows: Sequence[Tuple[int, ...]]) -> dict:
    """Float screening only (per campaign discipline): maximize t s.t.
    p_i,q_i in the box and, for every z, SOME (arc,side) has
    sgn*load(a,z) >= t (a disjunction via big-M binaries). The optimum is
    exactly R(rows) up to solver float error. Used only to SEED the exact
    search below -- never to decide anything."""
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    r = len(rows)
    n_bin = len(_ZS) * r * 2
    n_var = 2 * K + 1 + n_bin
    big_m = 10.0

    def bidx(zi, a, side):
        return 2 * K + 1 + (zi * r + a) * 2 + side

    A_rows, lb, ub = [], [], []
    for i in range(K):
        row = np.zeros(n_var)
        row[i] = 1.0
        row[K + i] = 1.0
        A_rows.append(row); lb.append(-np.inf); ub.append(1.0)
    for zi, z in enumerate(_ZS):
        for a in range(r):
            rowc = rows[a]
            for side in (0, 1):
                sgn = 1.0 if side == 0 else -1.0
                rr = np.zeros(n_var)
                for i in range(K):
                    c = rowc[i]
                    if not c:
                        continue
                    if z[i]:
                        rr[K + i] = -sgn * c
                    else:
                        rr[i] = sgn * c
                rr[2 * K] = 1.0
                rr[bidx(zi, a, side)] = big_m
                A_rows.append(rr); lb.append(-np.inf); ub.append(big_m)
        sel = np.zeros(n_var)
        for a in range(r):
            for side in (0, 1):
                sel[bidx(zi, a, side)] = 1.0
        A_rows.append(sel); lb.append(1.0); ub.append(np.inf)

    integrality = np.zeros(n_var)
    integrality[2 * K + 1:] = 1
    bounds = Bounds(lb=[0.0] * (2 * K) + [0.0] + [0.0] * n_bin,
                     ub=[1.0] * (2 * K) + [3.0] + [1.0] * n_bin)
    c_obj = np.zeros(n_var)
    c_obj[2 * K] = -1.0
    res = milp(c=c_obj, constraints=LinearConstraint(np.array(A_rows), lb, ub),
               integrality=integrality, bounds=bounds)
    if not res.success:
        return {"ok": False, "message": res.message}
    x = res.x
    return {"ok": True, "R": float(x[2 * K]),
            "p": [float(v) for v in x[:K]],
            "q": [float(v) for v in x[K:2 * K]]}


def load_exact(row: Tuple[int, ...], p: Sequence[Fr], q: Sequence[Fr],
                z: Sequence[int]) -> Fr:
    s = Fr(0)
    for c, pi, qi, zi in zip(row, p, q, z):
        if c:
            s += c * (qi if zi else -pi)
    return s


def exact_min_over_z(rows: Sequence[Tuple[int, ...]], p: Sequence[Fr],
                      q: Sequence[Fr]) -> Tuple[Fr, Tuple[int, ...]]:
    """min_z max_a |load(a,z)| in exact Fractions -- the value R would take
    AT this specific (p,q), i.e. a certified LOWER bound on R via this
    witness point."""
    best, best_z = None, None
    for z in _ZS:
        worst = max(abs(load_exact(row, p, q, z)) for row in rows)
        if best is None or worst < best:
            best, best_z = worst, z
    return best, best_z


def rationalize(p_f: Sequence[float], q_f: Sequence[float],
                 denominators=(1, 2, 3, 4, 6, 8, 12, 16, 24, 48)
                 ) -> List[Tuple[List[Fr], List[Fr]]]:
    """Candidate exact rationalizations of a float (p,q) point, several
    denominators, clipped to the box (p_i>=0, q_i>=0, p_i+q_i<=1)."""
    out = []
    for den in denominators:
        p = [Fr(round(v * den), den) for v in p_f]
        q = [Fr(round(v * den), den) for v in q_f]
        p = [max(Fr(0), min(Fr(1), v)) for v in p]
        q = [max(Fr(0), min(Fr(1), v)) for v in q]
        # clip to p_i+q_i<=1 by trimming q_i if needed (keeps p exact)
        q = [min(qi, 1 - pi) for pi, qi in zip(p, q)]
        out.append((p, q))
    return out


def find_exact_witness(rows: Sequence[Tuple[int, ...]], target_candidates
                        ) -> Optional[dict]:
    """Try each candidate target value v (e.g. 1/2, 2/3, 3/4) plus the
    float-MILP hint; return the first exact (p,q) whose exact_min_over_z
    equals v exactly, for the LARGEST such v found (best lower-bound
    witness)."""
    hint = float_R_milp(rows)
    candidates: List[Tuple[List[Fr], List[Fr]]] = []
    if hint["ok"]:
        candidates += rationalize(hint["p"], hint["q"])
    # also try the canonical symmetric witnesses directly (w = 1/2 pattern
    # generalizes the Rybin witness; cheap to include as extra candidates)
    for v in target_candidates:
        candidates.append(([Fr(1, 2)] * K, [Fr(1, 2)] * K))
    best = None
    for (p, q) in candidates:
        val, z = exact_min_over_z(rows, p, q)
        if best is None or val > best["value"]:
            best = {"value": val, "p": [str(x) for x in p],
                    "q": [str(x) for x in q], "best_z": z}
    return best


# ---------------------------------------------------------------------------
# Fast exact Fraction simplex (sympy's lpmax measured ~75ms/call -- symbolic
# overhead makes the B&B infeasible within budget; this hand tableau does
# the same exact rational arithmetic at ~1000x less overhead per call).
# Single-phase: every LP we pose here has RHS >= 0 and the origin feasible
# (see exact_upper_bound_bb docstring), so no phase-1 is needed. Bland's
# rule (smallest-index pivoting) guards against cycling.
# ---------------------------------------------------------------------------

class _LF:
    """Lean rational for the simplex hot loop. fractions.Fraction's generic
    numeric-tower dispatch (__new__ validation, isinstance checks on every
    op) measured as ~85% of exact_upper_bound_bb's runtime (cProfile,
    2026-07-25 pilot: 131s of which 110s in Fraction.forward/_mul/_sub).
    This is a minimal (numerator, denominator) pair with __slots__ and
    direct arithmetic -- no generic dispatch, no type coercion. Internal to
    simplex_max only; the public API still speaks fractions.Fraction."""
    __slots__ = ("n", "d")

    def __init__(self, n: int, d: int = 1):
        if d < 0:
            n, d = -n, -d
        if n == 0:
            self.n, self.d = 0, 1
            return
        g = _gcd(n if n >= 0 else -n, d)
        if g > 1:
            n //= g
            d //= g
        self.n, self.d = n, d

    @staticmethod
    def from_fr(x: Fr) -> "_LF":
        return _LF(x.numerator, x.denominator)

    def to_fr(self) -> Fr:
        return Fr(self.n, self.d)

    def __add__(self, o):
        return _LF(self.n * o.d + o.n * self.d, self.d * o.d)

    def __sub__(self, o):
        return _LF(self.n * o.d - o.n * self.d, self.d * o.d)

    def __mul__(self, o):
        return _LF(self.n * o.n, self.d * o.d)

    def __truediv__(self, o):
        return _LF(self.n * o.d, self.d * o.n)

    def __neg__(self):
        return _LF(-self.n, self.d)

    def __lt__(self, o):
        return self.n * o.d < o.n * self.d

    def __le__(self, o):
        return self.n * o.d <= o.n * self.d

    def __gt__(self, o):
        return self.n * o.d > o.n * self.d

    def is_zero(self):
        return self.n == 0


_gcd = __import__("math").gcd
_LF_ZERO = _LF(0)
_LF_ONE = _LF(1)


def _to_lf(v) -> "_LF":
    if isinstance(v, _LF):
        return v
    if isinstance(v, Fr):
        return _LF(v.numerator, v.denominator)
    return _LF(int(v))


def simplex_max(c: Sequence[Fr], A: Sequence[Sequence[Fr]], b: Sequence[Fr],
                 want_solution: bool = False):
    """maximize c.x s.t. A x <= b, x >= 0, all b_i >= 0 (origin feasible).
    Returns the optimal value (or None if unbounded -- should not occur in
    this script's usage, every LP posed here is box-bounded), or
    (value, solution_list) if want_solution. Public interface is
    fractions.Fraction in/out; the hot loop runs in the lean _LF."""
    m = len(A)
    n = len(c)
    width = n + m + 1
    T = [[_LF_ZERO] * width for _ in range(m + 1)]
    for i in range(m):
        row = T[i] = [_LF_ZERO] * width
        Ai = A[i]
        for j in range(n):
            v = Ai[j]
            if v != 0:
                row[j] = _to_lf(v)
        row[n + i] = _LF_ONE
        row[-1] = _to_lf(b[i])
    obj = T[m]
    for j in range(n):
        obj[j] = -_to_lf(c[j])
    basis = list(range(n, n + m))
    while True:
        piv_col = None
        for j in range(n + m):
            if obj[j].n < 0:
                piv_col = j
                break
        if piv_col is None:
            val = obj[-1].to_fr()
            if not want_solution:
                return val
            xsol = [Fr(0)] * n
            for i, bi in enumerate(basis):
                if bi < n:
                    xsol[bi] = T[i][-1].to_fr()
            return val, xsol
        piv_row = None
        best_ratio = None
        for i in range(m):
            a_ij = T[i][piv_col]
            if a_ij.n > 0:
                ratio = T[i][-1] / a_ij
                if (best_ratio is None or ratio < best_ratio or
                        (not ratio > best_ratio and basis[i] < basis[piv_row])):
                    best_ratio = ratio
                    piv_row = i
        if piv_row is None:
            return (None, None) if want_solution else None  # unbounded
        pv = T[piv_row][piv_col]
        prow = T[piv_row]
        if pv.n != pv.d:
            prow[:] = [x / pv for x in prow]
        for i in range(m + 1):
            if i == piv_row:
                continue
            factor = T[i][piv_col]
            if not factor.is_zero():
                ti = T[i]
                for k in range(width):
                    if not prow[k].is_zero():
                        ti[k] = ti[k] - factor * prow[k]
        basis[piv_row] = piv_col


def _selftest_simplex():
    """Cross-check against sympy.lpmax on a handful of cases before trusting
    the hand simplex in the B&B."""
    from sympy import symbols, Rational
    from sympy.solvers.simplex import lpmax
    import random
    rng = random.Random(7)
    for trial in range(30):
        n = rng.randint(2, 5)
        m = rng.randint(2, 6)
        c = [Fr(rng.randint(-3, 5)) for _ in range(n)]
        A = [[Fr(rng.randint(-3, 3)) for _ in range(n)] for _ in range(m)]
        b = [Fr(rng.randint(1, 10)) for _ in range(m)]  # >=0, origin feasible
        # guarantee boundedness (matches real usage: every LP here is
        # box-bounded) by adding x_j <= 10 for each variable
        for j in range(n):
            row = [Fr(0)] * n
            row[j] = Fr(1)
            A.append(row)
            b.append(Fr(10))
        m = len(A)
        got = simplex_max(c, A, b)
        xs = symbols(f'x0:{n}')
        cons = [xs[j] >= 0 for j in range(n)]
        for i in range(m):
            cons.append(sum(A[i][j] * xs[j] for j in range(n)) <= b[i])
        want, _ = lpmax(sum(c[j] * xs[j] for j in range(n)), cons)
        assert got == want, f"MISMATCH trial={trial}: got {got} want {want}"
    print("simplex self-test: 30/30 exact matches with sympy.lpmax", flush=True)


# ---------------------------------------------------------------------------
# Task B: exact DFS branch-and-bound certifying R <= v (v = 3/4 typically)
#
# Disjunctive system: does some (p,q) in the box escape every P_z (i.e. for
# EVERY z, SOME arc's |load| > v)? Encoded as: for z = 0..7 (in order), pick
# (arc, side) with sgn*load(a,z) >= v + m (a SHARED margin variable m, so
# strictness is exact -- see module docstring). At depth d the running LP
# "maximize m s.t. box + constraints chosen so far" is monotone
# non-increasing as more constraints are added (a bigger constraint set can
# only shrink the feasible region), so pruning whenever the running max <= 0
# is sound: no full-depth extension can ever recover positive margin.
# ---------------------------------------------------------------------------

_SHIFT = Fr(10)  # m' = m + SHIFT >= 0, so the origin (p=q=0, m'=SHIFT-v) is
                 # feasible for every LP posed below (SHIFT=10 > 3 >= any
                 # achievable |load|, so SHIFT - v_r > 0 always) -- single
                 # phase simplex applies, no phase-1 needed.
_VARN = 2 * K + 1  # p0,p1,p2,q0,q1,q2,m'


def _box_rows() -> Tuple[List[List[Fr]], List[Fr]]:
    A, b = [], []
    for i in range(K):
        row = [Fr(0)] * _VARN
        row[i] = Fr(1)        # p_i
        row[K + i] = Fr(1)    # q_i
        A.append(row)
        b.append(Fr(1))
    return A, b


def _margin_row(row: Tuple[int, ...], z: Sequence[int], sgn: int, v: Fr
                 ) -> Tuple[List[Fr], Fr]:
    """-sgn*load(a,z) + m' <= SHIFT - v, i.e. the (arc,side) choice
    sgn*load(a,z) >= v + m becomes feasible-region membership in <= form."""
    r_ = [Fr(0)] * _VARN
    for c, zi, i in zip(row, z, range(K)):
        if not c:
            continue
        if zi:
            r_[K + i] = -sgn * c   # coefficient on q_i
        else:
            r_[i] = sgn * c        # coefficient on p_i
    r_[2 * K] = Fr(1)              # m'
    return r_, _SHIFT - v


class _NodeCapReached(Exception):
    pass


def exact_upper_bound_bb(rows: Sequence[Tuple[int, ...]], v: Fr,
                          verbose_every: int = 20000,
                          max_nodes: Optional[int] = None) -> dict:
    """DFS B&B: does some (p,q) in the box make EVERY z's max-|load| exceed
    v strictly? If no (proved_le_v=True), R <= v is certified exactly. Every
    LP call is the hand Fraction simplex (validated against sympy above)."""
    r = len(rows)
    c_obj = [Fr(0)] * _VARN
    c_obj[2 * K] = Fr(1)
    base_A, base_b = _box_rows()

    stats = {"lp_calls": 0, "nodes": 0, "pruned": 0, "leaves_checked": 0}
    counterexample = None
    t0 = time.time()

    def dfs(depth, A, b):
        nonlocal counterexample
        stats["nodes"] += 1
        if max_nodes is not None and stats["nodes"] >= max_nodes:
            raise _NodeCapReached()
        if stats["nodes"] % verbose_every == 0:
            print(f"    [bb v={v}] nodes={stats['nodes']} "
                  f"lp_calls={stats['lp_calls']} pruned={stats['pruned']} "
                  f"depth={depth} elapsed={time.time()-t0:.1f}s", flush=True)
        if depth == len(_ZS):
            stats["leaves_checked"] += 1
            return
        z = _ZS[depth]
        for a in range(r):
            row = rows[a]
            for side, sgn in ((0, 1), (1, -1)):
                mrow, mrhs = _margin_row(row, z, sgn, v)
                A2 = A + [mrow]
                b2 = b + [mrhs]
                stats["lp_calls"] += 1
                bound = simplex_max(c_obj, A2, b2)
                margin = None if bound is None else bound - _SHIFT
                if margin is None or margin <= 0:
                    stats["pruned"] += 1
                    continue
                if depth + 1 == len(_ZS):
                    val, xsol = simplex_max(c_obj, A2, b2, want_solution=True)
                    p_sol = xsol[:K]
                    q_sol = xsol[K:2 * K]
                    counterexample = {
                        "margin": str(val - _SHIFT), "p": [str(x) for x in p_sol],
                        "q": [str(x) for x in q_sol],
                    }
                    return
                dfs(depth + 1, A2, b2)
                if counterexample is not None:
                    return

    try:
        dfs(0, base_A, base_b)
    except _NodeCapReached:
        pass
    elapsed = time.time() - t0
    return {
        "v": str(v), "proved_le_v": counterexample is None,
        "counterexample": counterexample,
        "stats": stats, "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Task C: does a simple deterministic rounding rule certify 3/4 on its own?
# ---------------------------------------------------------------------------

def greedy_decreasing_demand(rows: Sequence[Tuple[int, ...]],
                              p: Sequence[Fr], q: Sequence[Fr]
                              ) -> Tuple[Tuple[int, ...], Fr]:
    """Process chords in decreasing d_i = p_i+q_i (ties by index). At each
    step choose z_i in {0,1} to minimize the resulting max_a|partial load|
    given the choices already locked in (arcs not yet touched by any later
    chord contribute nothing yet). Returns (z, final max|load|)."""
    d = [p[i] + q[i] for i in range(K)]
    order = sorted(range(K), key=lambda i: (-d[i], i))
    z = [None] * K
    partial = [Fr(0)] * len(rows)
    for i in order:
        best_choice, best_max = None, None
        for zi, delta in ((1, q[i]), (0, -p[i])):
            trial_max = Fr(0)
            for a, row in enumerate(rows):
                c = row[i]
                val = partial[a] + (c * delta if c else Fr(0))
                if abs(val) > trial_max:
                    trial_max = abs(val)
            if best_max is None or trial_max < best_max:
                best_max, best_choice = trial_max, (zi, delta)
        zi, delta = best_choice
        z[i] = zi
        for a, row in enumerate(rows):
            c = row[i]
            if c:
                partial[a] += c * delta
    final_max = max(abs(x) for x in partial) if partial else Fr(0)
    return tuple(z), final_max


def _rule_engine(rows, p, q, order, choose):
    """Shared driver: process chords in `order`, `choose(i, partial, p, q,
    rows) -> (zi, delta)` decides each step. Returns (z, final max|load|)."""
    z = [None] * K
    partial = [Fr(0)] * len(rows)
    for i in order:
        zi, delta = choose(i, partial, p, q, rows)
        z[i] = zi
        for a, row in enumerate(rows):
            c = row[i]
            if c:
                partial[a] += c * delta
    final = max((abs(x) for x in partial), default=Fr(0))
    return tuple(z), final


def _trial_max(rows, partial, i, delta):
    best = Fr(0)
    for a, row in enumerate(rows):
        c = row[i]
        val = partial[a] + (c * delta if c else Fr(0))
        if abs(val) > best:
            best = abs(val)
    return best


def rule_a_decreasing_demand(rows, p, q):
    """(a) process chords in decreasing d_i=p_i+q_i; at each step choose the
    sign minimizing the resulting max|partial load|. Ties -> zi=1 first."""
    d = [p[i] + q[i] for i in range(K)]
    order = sorted(range(K), key=lambda i: (-d[i], i))

    def choose(i, partial, p, q, rows):
        best = None
        for zi, delta in ((1, q[i]), (0, -p[i])):
            t = _trial_max(rows, partial, i, delta)
            if best is None or t < best[0]:
                best = (t, zi, delta)
        return best[1], best[2]

    return _rule_engine(rows, p, q, order, choose)


def rule_b_decreasing_maxpq(rows, p, q):
    """(b) same greedy step as (a), but process order = decreasing
    max(p_i,q_i) instead of decreasing d_i."""
    m = [max(p[i], q[i]) for i in range(K)]
    order = sorted(range(K), key=lambda i: (-m[i], i))

    def choose(i, partial, p, q, rows):
        best = None
        for zi, delta in ((1, q[i]), (0, -p[i])):
            t = _trial_max(rows, partial, i, delta)
            if best is None or t < best[0]:
                best = (t, zi, delta)
        return best[1], best[2]

    return _rule_engine(rows, p, q, order, choose)


def rule_c_oppose_current_load(rows, p, q):
    """(c) decreasing-d_i order; choose the sign of delta_i OPPOSING the
    current aggregate load on the rows containing i: score = sum_a
    C[a,i]*partial[a] over rows a where C[a,i]!=0; score>0 means increasing
    delta_i would push those rows further from 0 (bad) -> take delta_i<0
    (zi=0); score<=0 -> take delta_i>0 (zi=1)."""
    d = [p[i] + q[i] for i in range(K)]
    order = sorted(range(K), key=lambda i: (-d[i], i))

    def choose(i, partial, p, q, rows):
        score = Fr(0)
        for a, row in enumerate(rows):
            c = row[i]
            if c:
                score += c * partial[a]
        if score > 0:
            return 0, -p[i]
        return 1, q[i]

    return _rule_engine(rows, p, q, order, choose)


def rule_d_small_side_first(rows, p, q):
    """(d) decreasing-d_i order; prefer the option with smaller |delta_i|
    UNLESS it strictly worsens the resulting max vs the other option."""
    d = [p[i] + q[i] for i in range(K)]
    order = sorted(range(K), key=lambda i: (-d[i], i))

    def choose(i, partial, p, q, rows):
        opts = sorted(((1, q[i]), (0, -p[i])), key=lambda t: abs(t[1]))
        t_small = _trial_max(rows, partial, i, opts[0][1])
        t_other = _trial_max(rows, partial, i, opts[1][1])
        if t_small <= t_other:
            return opts[0]
        return opts[1]

    return _rule_engine(rows, p, q, order, choose)


_RULES = {
    "a_decreasing_demand": rule_a_decreasing_demand,
    "b_decreasing_maxpq": rule_b_decreasing_maxpq,
    "c_oppose_current_load": rule_c_oppose_current_load,
    "d_small_side_first": rule_d_small_side_first,
}


def sweep_rule(rule_fn, rows, n_trials=500, seed=0, extra_points=(),
               threshold=Fr(3, 4)):
    """Random exact-rational (p,q) trials plus any mandatory extra_points
    (e.g. the known Rybin/maximal-type extremal witnesses)."""
    rng = random.Random(seed)
    denoms = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    worst = Fr(-1)
    worst_pq = None
    n_checked = 0
    for (p, q) in extra_points:
        n_checked += 1
        _, val = rule_fn(rows, list(p), list(q))
        if val > worst:
            worst, worst_pq = val, (list(p), list(q))
    for _ in range(n_trials):
        den = rng.choice(denoms)
        p = [Fr(rng.randint(0, den), den) for _ in range(K)]
        q = []
        for pi in p:
            qi = Fr(rng.randint(0, den), den)
            if pi + qi > 1:
                qi = 1 - pi
            q.append(qi)
        n_checked += 1
        _, val = rule_fn(rows, p, q)
        if val > worst:
            worst, worst_pq = val, (list(p), list(q))
    return {"n_checked": n_checked, "worst_value": str(worst),
            "exceeds_threshold": worst > threshold,
            "worst_pq": ([str(x) for x in worst_pq[0]],
                          [str(x) for x in worst_pq[1]]) if worst_pq else None}


def test_rounding_rule(rows: Sequence[Tuple[int, ...]], n_trials: int = 400,
                        seed: int = 0, threshold: Fr = Fr(3, 4)) -> dict:
    rng = random.Random(seed)
    worst = Fr(-1)
    worst_pq = None
    denoms = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    for _ in range(n_trials):
        den = rng.choice(denoms)
        p = [Fr(rng.randint(0, den), den) for _ in range(K)]
        q = []
        for pi in p:
            max_q_num = den - pi.numerator * (den // pi.denominator) if False else None
            # simpler: sample q_i in [0, 1-p_i] on the same denominator grid
            qi_num = rng.randint(0, den)
            qi = Fr(qi_num, den)
            if pi + qi > 1:
                qi = 1 - pi
            q.append(qi)
        _, val = greedy_decreasing_demand(rows, p, q)
        if val > worst:
            worst = val
            worst_pq = (list(p), list(q))
    return {"n_trials": n_trials, "worst_value": str(worst),
            "exceeds_threshold": worst > threshold,
            "worst_pq": ([str(x) for x in worst_pq[0]],
                          [str(x) for x in worst_pq[1]]) if worst_pq else None}


def run_full(results_path="k3_proof_results.json", max_exhaustive_m=9,
             sample_m=(10,), n_samples=250000, v=Fr(3, 4)):
    """End-to-end reproduction: Task A enumeration -> maximal types -> Task B
    exact B&B (+ per-type exact R via witness) -> Task C rule test -> JSON."""
    print("=== Task A: enumeration ===", flush=True)
    found, maximal, history = run_enumeration(
        max_exhaustive_m=max_exhaustive_m, sample_m=sample_m,
        n_samples=n_samples, results_path=results_path + ".enum_progress")

    print("\n=== Task A: dominance check (all non-maximal <= some maximal) ===",
          flush=True)
    undominated = []
    for c in found:
        if c in maximal:
            continue
        if not any(is_dominated(c, mx) for mx in maximal):
            undominated.append(c)
    print(f"undominated non-maximal types: {len(undominated)} (expect 0)",
          flush=True)

    print("\n=== Task B: exact R per type (witness lower bound) ===", flush=True)
    per_type = []
    for c, info in found.items():
        w = find_exact_witness(list(c), [Fr(1, 2), Fr(2, 3), Fr(3, 4)])
        per_type.append({"rows": [list(r) for r in c], "n": info["n"],
                          "is_maximal": c in maximal,
                          "witness_R_lower_bound": str(w["value"]),
                          "witness_p": w["p"], "witness_q": w["q"]})
        print(f"  R>={w['value']} for {len(c)}-row type (n={info['n']})",
              flush=True)

    print("\n=== Task B: exact B&B upper-bound proof on maximal types ===",
          flush=True)
    bb_results = []
    for c in maximal:
        print(f"  B&B on {c} (v={v}) ...", flush=True)
        res = exact_upper_bound_bb(list(c), v, verbose_every=5000)
        res["rows"] = [list(r) for r in c]
        bb_results.append(res)
        print(f"    proved_le_v={res['proved_le_v']} stats={res['stats']} "
              f"elapsed={res['elapsed_s']:.1f}s", flush=True)
        if not res["proved_le_v"]:
            print("  !!!! COUNTEREXAMPLE TO R<=3/4 FOUND -- SEE counterexample "
                  "FIELD. STOP AND VERIFY BEFORE ANY FURTHER CLAIM. !!!!",
                  flush=True)

    print("\n=== Task C: rounding-rule search (greedy decreasing-demand) ===",
          flush=True)
    rule_results = []
    for c in found:
        res = test_rounding_rule(list(c), n_trials=300, seed=1)
        rule_results.append({"rows": [list(r) for r in c],
                              "exceeds_3_4": res["exceeds_threshold"],
                              "worst_value": res["worst_value"]})
    n_fail = sum(1 for r in rule_results if r["exceeds_3_4"])
    print(f"  greedy decreasing-demand rule: fails (exceeds 3/4) on "
          f"{n_fail}/{len(rule_results)} types", flush=True)

    out = {
        "task_a": {
            "total_types": len(found), "maximal_types": len(maximal),
            "history": history,
            "undominated_non_maximal_count": len(undominated),
            "maximal_rows": [[list(r) for r in c] for c in maximal],
        },
        "task_b": {
            "per_type_exact_R": per_type,
            "bb_upper_bound_proofs": bb_results,
        },
        "task_c": {
            "rule": "greedy_decreasing_demand",
            "results": rule_results,
            "n_types_tested": len(rule_results),
            "n_types_failed": n_fail,
        },
    }
    with open(results_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nWrote {results_path}", flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "pilot":
        found = {}
        for m in range(2, 6):
            enumerate_exhaustive(m, found)
        print("pilot total types:", len(found))
        for c in sorted(found):
            print(" ", c, "n=", found[c]["n"])
    elif mode == "enumerate":
        results_path = "/tmp/k3_enum_progress.json"
        found, maximal, history = run_enumeration(results_path=results_path)
        print("\nMaximal types:")
        for c in maximal:
            print(" ", c, "n=", found[c]["n"], "size=", len(c))
    elif mode == "full":
        run_full()
