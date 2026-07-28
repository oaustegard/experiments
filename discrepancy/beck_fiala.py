"""Beck-Fiala small-t exact-value engine (issue #166, Target B).

D(t, n) = max over set systems on n elements with element-degree <= t of the
combinatorial discrepancy  disc(S) = min_{x in {+-1}^n} max_{S_i} |sum_{j in
S_i} x_j|.

Key WLOG: a degree-<=t system on n elements has at most t*n nonempty sets
(each set uses >= 1 of the <= t*n incidences), and duplicate/empty sets never
change disc.  So fixing m = t*n rows (empty rows allowed) makes the CEGAR
search exact for D(t, n), not just a lower bound at some m.

Engines:
* disc_exact       — exhaustive over 2^{n-1} colorings, integer arithmetic.
* sat_disc_geq     — certificate that a FIXED system has disc >= k: encode
                     "exists coloring with all |imbalances| <= k-1" and get
                     UNSAT (cross-checked against disc_exact).
* cegar_exists     — decide "exists degree-<=t system on n elements with
                     disc >= k": SAT over incidence-matrix variables with
                     counterexample-guided coloring constraints.
"""

from __future__ import annotations

import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver

from komlos import all_sign_vectors

SOLVER_NAME = "cadical153"


# ---------------------------------------------------------------------------
# Exact discrepancy of a fixed 0/1 incidence matrix (rows = sets)
# ---------------------------------------------------------------------------

def disc_exact(M: np.ndarray) -> tuple[int, np.ndarray]:
    """Exhaustive exact disc; returns (disc, argmin coloring)."""
    M = np.asarray(M, dtype=np.int64)
    if M.size == 0 or M.sum() == 0:
        return 0, np.ones(M.shape[1] if M.ndim == 2 else 0, dtype=np.int64)
    n = M.shape[1]
    S = all_sign_vectors(n)
    E = S @ M.T
    maxima = np.abs(E).max(axis=1)
    idx = int(maxima.argmin())
    return int(maxima[idx]), S[idx]


def max_degree(M: np.ndarray) -> int:
    return int(np.asarray(M).sum(axis=0).max()) if np.asarray(M).size else 0


# ---------------------------------------------------------------------------
# SAT certificate: fixed system has disc >= k
# ---------------------------------------------------------------------------

def _coloring_clauses(M: np.ndarray, k: int, pool: IDPool) -> list[list[int]]:
    """CNF for: exists x in {+-1}^n with |imbalance(S_i)| <= k-1 for all i.

    Variable x_j true  <=> element j colored +1.  For set S with |S| = c and
    p = #(+1 in S): imbalance = 2p - c, so |imb| <= k-1  <=>
    ceil((c-k+1)/2) <= p <= floor((c+k-1)/2).
    """
    clauses: list[list[int]] = []
    n = M.shape[1]
    xs = [pool.id(("x", j)) for j in range(n)]
    for i in range(M.shape[0]):
        members = [xs[j] for j in range(n) if M[i, j]]
        c = len(members)
        if c == 0:
            continue
        lo = max(0, -((-(c - k + 1)) // 2))  # ceil((c-k+1)/2), floored at 0
        hi = min(c, (c + k - 1) // 2)
        if lo > 0:
            enc = CardEnc.atleast(lits=members, bound=lo, vpool=pool,
                                  encoding=EncType.seqcounter)
            clauses += enc.clauses
        if hi < c:
            enc = CardEnc.atmost(lits=members, bound=hi, vpool=pool,
                                 encoding=EncType.seqcounter)
            clauses += enc.clauses
    return clauses


def sat_disc_geq(M: np.ndarray, k: int) -> bool:
    """True iff disc(M) >= k, decided by SAT (UNSAT of the <= k-1 coloring)."""
    M = np.asarray(M, dtype=np.int64)
    pool = IDPool()
    clauses = _coloring_clauses(M, k, pool)
    with Solver(name=SOLVER_NAME, bootstrap_with=clauses) as s:
        return not s.solve()


# ---------------------------------------------------------------------------
# CEGAR: exists degree-<=t system on n elements (m rows) with disc >= k ?
# ---------------------------------------------------------------------------

def _lex_ge_rows(pool: IDPool, avars: list[list[int]]) -> list[list[int]]:
    """Symmetry breaking: row i >=_lex row i+1 (rows are freely permutable).

    Standard prefix-equality encoding: e_{i,j} = rows i, i+1 agree on columns
    < j.  Sound because every constraint in the CEGAR model is symmetric
    under row permutation (degree sums are per column; the counterexample
    disjunctions range over all rows).
    """
    clauses: list[list[int]] = []
    m = len(avars)
    n = len(avars[0]) if m else 0
    for i in range(m - 1):
        top, bot = avars[i], avars[i + 1]
        e_prev = None
        for j in range(n):
            if e_prev is None:
                # column 0: forbid top[0] < bot[0]
                clauses.append([top[0], -bot[0]])
            else:
                # e_prev -> not(top[j] < bot[j])
                clauses.append([-e_prev, top[j], -bot[j]])
            if j < n - 1:
                e = pool.id(("lex_e", i, j))
                if e_prev is None:
                    # e <-> (top[0] == bot[0])
                    clauses += [[-e, top[0], -bot[0]], [-e, -top[0], bot[0]],
                                [e, top[0], bot[0]], [e, -top[0], -bot[0]]]
                else:
                    # e <-> e_prev & (top[j] == bot[j])
                    clauses += [[-e, e_prev],
                                [-e, top[j], -bot[j]], [-e, -top[j], bot[j]],
                                [e, -e_prev, top[j], bot[j]],
                                [e, -e_prev, -top[j], -bot[j]]]
                e_prev = e
    return clauses


def _imbalance_indicator(pool: IDPool, row_vars: list[int], x: np.ndarray,
                         k: int, z: int, positive: bool) -> list[list[int]]:
    """Clauses for  z -> (imbalance of this row under coloring x >= k)
    (or <= -k when positive=False).

    With P = {j: x_j = +1}, N = {j: x_j = -1}, s = sum_P a_j, d = sum_N a_j:
      s - d >= k  <=>  sum_P a_j + sum_N (not a_j) >= k + |N|
      d - s >= k  <=>  sum_N a_j + sum_P (not a_j) >= k + |P|
    Conditioning on z: append -z to every clause of the cardinality encoding
    (aux variables are fresh per call, so relaxed encodings stay sound).
    """
    P = [j for j in range(len(x)) if x[j] > 0]
    N = [j for j in range(len(x)) if x[j] < 0]
    if positive:
        lits = [row_vars[j] for j in P] + [-row_vars[j] for j in N]
        bound = k + len(N)
    else:
        lits = [row_vars[j] for j in N] + [-row_vars[j] for j in P]
        bound = k + len(P)
    if bound > len(lits):
        return [[-z]]  # impossible: forbid the indicator
    enc = CardEnc.atleast(lits=lits, bound=bound, vpool=pool,
                          encoding=EncType.seqcounter)
    return [cl + [-z] for cl in enc.clauses]


def cegar_exists(t: int, n: int, k: int, m: int | None = None,
                 max_rounds: int = 4000, verbose: bool = False,
                 lex_break: bool = True,
                 conf_budget: int | None = None) -> dict:
    """Decide: does a set system on n elements with element-degree <= t and
    m rows (default t*n, the WLOG-complete choice) have disc >= k?

    Returns {"exists": bool | None, "witness": M or None, "rounds": r}.
    "exists" False is exact for m = t*n: D(t, n) < k.
    "exists" None means the per-call SAT conflict budget ran out before a
    verdict — the answer is UNKNOWN, not a bound.
    """
    if m is None:
        m = t * n
    pool = IDPool()
    avars = [[pool.id(("a", i, j)) for j in range(n)] for i in range(m)]
    base: list[list[int]] = []
    for j in range(n):
        enc = CardEnc.atmost(lits=[avars[i][j] for i in range(m)], bound=t,
                             vpool=pool, encoding=EncType.seqcounter)
        base += enc.clauses
    if lex_break:
        base += _lex_ge_rows(pool, avars)

    with Solver(name=SOLVER_NAME, bootstrap_with=base) as s:
        for rounds in range(1, max_rounds + 1):
            if conf_budget is not None:
                s.conf_budget(conf_budget)
                verdict = s.solve_limited()
                if verdict is None:
                    return {"exists": None, "witness": None, "rounds": rounds}
            else:
                verdict = s.solve()
            if not verdict:
                return {"exists": False, "witness": None, "rounds": rounds}
            model = set(l for l in s.get_model() if l > 0)
            M = np.array([[1 if avars[i][j] in model else 0
                           for j in range(n)] for i in range(m)],
                         dtype=np.int64)
            d, x = disc_exact(M)
            if d >= k:
                return {"exists": True, "witness": M, "rounds": rounds}
            # counterexample: coloring x keeps every |imbalance| <= k-1;
            # require some row to break x (in either direction)
            or_lits = []
            for i in range(m):
                zp = pool.id(("zp", rounds, i))
                zm = pool.id(("zm", rounds, i))
                for cl in _imbalance_indicator(pool, avars[i], x, k, zp, True):
                    s.add_clause(cl)
                for cl in _imbalance_indicator(pool, avars[i], x, k, zm, False):
                    s.add_clause(cl)
                or_lits += [zp, zm]
            s.add_clause(or_lits)
            if verbose and rounds % 25 == 0:
                print(f"  cegar t={t} n={n} k={k}: round {rounds}, "
                      f"cand disc {d}")
    raise RuntimeError(f"CEGAR did not converge in {max_rounds} rounds "
                       f"(t={t}, n={n}, k={k})")


def canonical_system(M: np.ndarray) -> list[str]:
    """Human-readable sorted nonempty sets, e.g. ['{0,1,2}', ...]."""
    out = []
    for row in np.asarray(M):
        members = [str(j) for j in range(len(row)) if row[j]]
        if members:
            out.append("{" + ",".join(members) + "}")
    return sorted(out)
