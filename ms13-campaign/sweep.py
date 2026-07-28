"""
ms13-campaign sweep.py: float-search SCREENING engine (claude-workspace
issue #169). This module never makes a decision -- every "found a
counterexample" claim is confirmed by an exact re-evaluation through
core.verify_counterexample before it is reported as such.

Search strategy: sample random points on the product of per-terminal path
simplices (Dirichlet starts), track the float-best by t(x), then
coordinate-wise hill-climb ("polish") from that start. Floats are used
throughout the search for speed; the final candidate is rationalized
(several denominators tried) and every candidate is re-scored EXACTLY via
core.t_of_x/verify_counterexample, so the reported "exact" numbers never
depend on float arithmetic.

Anchor (prior session, random layered DAGs): rho = (d_max + t*)/d_max
plateaus around 0.84 -- no counterexamples from unstructured random search.
This sweep should qualitatively reproduce "plateau clearly below 1".
"""

from __future__ import annotations

import random
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import Arc, Instance, Terminal, check_paths_valid, verify_counterexample

# ---------------------------------------------------------------------------
# Exact wrapper (no floats touch this path)
# ---------------------------------------------------------------------------

def t_of_x(inst: Instance, split: Sequence[Sequence[Fr]]) -> Fr:
    """Exact t(x) = min_P max_a |f^P(a) - x(a)| - d_max for a given split.

    Thin wrapper over core.verify_counterexample -- the single source of
    truth for the two-sided violation objective.
    """
    return verify_counterexample(inst, split)["t_of_x"]


# ---------------------------------------------------------------------------
# Rationalization: float split -> several exact candidate splits
# ---------------------------------------------------------------------------

DEFAULT_DENOMS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 12, 16, 20, 30, 60, 200)


def _rationalize_one(ws: Sequence[float], denom: int) -> List[Fr]:
    fr = [Fr(w).limit_denominator(denom) for w in ws]
    s = sum(fr)
    if s == 0:
        return [Fr(1, len(ws))] * len(ws)
    # exact division: sum(fr) / s == 1 exactly, so this is always a valid
    # distribution regardless of how coarse `denom` was.
    return [f / s for f in fr]


def rationalize_split_candidates(
    split_float: Sequence[Sequence[float]],
    denoms: Sequence[int] = DEFAULT_DENOMS,
) -> List[List[List[Fr]]]:
    """One exact candidate split per denominator in `denoms`. The caller
    scores each with t_of_x and keeps the best -- a small denominator
    sometimes snaps onto the true rational optimum, but a coarse one can
    also wash out signal, so we don't guess which will win.
    """
    return [[_rationalize_one(ws, d) for ws in split_float] for d in denoms]


def best_rational_split(
    inst: Instance, split_float: Sequence[Sequence[float]],
    denoms: Sequence[int] = DEFAULT_DENOMS,
) -> Tuple[List[List[Fr]], Fr]:
    """Rationalize `split_float` several ways, score each exactly, return the
    (split, t_of_x) pair with the largest exact t.
    """
    best_split: Optional[List[List[Fr]]] = None
    best_t: Optional[Fr] = None
    for cand in rationalize_split_candidates(split_float, denoms):
        t_exact = t_of_x(inst, cand)
        if best_t is None or t_exact > best_t:
            best_t, best_split = t_exact, cand
    assert best_split is not None and best_t is not None
    return best_split, best_t


# ---------------------------------------------------------------------------
# Float-side precompute (numpy) for fast search
# ---------------------------------------------------------------------------

def _precompute_np(inst: Instance):
    arcs = inst.arcs
    routings = inst.routings()
    loads = np.array(
        [[float(inst.routing_load(r)[a]) for a in arcs] for r in routings],
        dtype=float,
    )
    dmax = float(inst.d_max())
    return arcs, routings, loads, dmax


def _precompute_incidence(inst: Instance, arcs: Sequence[Arc]) -> List[np.ndarray]:
    arc_index = {a: i for i, a in enumerate(arcs)}
    incidences = []
    for t in inst.terminals:
        d = float(t.demand)
        mat = np.zeros((len(t.paths), len(arcs)))
        for pi, path in enumerate(t.paths):
            for a in path:
                mat[pi, arc_index[a]] += d
        incidences.append(mat)
    return incidences


def _x_from_split_np(incidences: Sequence[np.ndarray], split: Sequence[np.ndarray]) -> np.ndarray:
    n_arcs = incidences[0].shape[1] if incidences else 0
    x = np.zeros(n_arcs)
    for mat, ws in zip(incidences, split):
        x += np.asarray(ws, dtype=float) @ mat
    return x


def _t_float(loads: np.ndarray, dmax: float, x_vec: np.ndarray) -> float:
    return float(np.min(np.max(np.abs(loads - x_vec), axis=1)) - dmax)


# ---------------------------------------------------------------------------
# screen_instance: the maximization heuristic
# ---------------------------------------------------------------------------

def screen_instance(inst: Instance, n_samples: int = 150, n_polish: int = 30,
                     seed: int = 0) -> Dict:
    """Maximize t(x) over the product of per-terminal path simplices.

    Float search (Dirichlet sampling + coordinate-wise hill-climb polish),
    then exact rationalization + re-scoring. Deterministic under `seed`.

    Returns a dict with both the float-best (search-only, not trustworthy on
    its own) and the exact-rational-best (the only number that can back a
    counterexample claim).
    """
    arcs, routings, loads, dmax = _precompute_np(inst)
    incidences = _precompute_incidence(inst, arcs)
    n_paths = [len(t.paths) for t in inst.terminals]
    rng = np.random.default_rng(seed)

    def random_split() -> List[np.ndarray]:
        return [rng.dirichlet(np.ones(n)) for n in n_paths]

    def score(split: Sequence[np.ndarray]) -> Tuple[float, np.ndarray]:
        x = _x_from_split_np(incidences, split)
        return _t_float(loads, dmax, x), x

    best_split: Optional[List[np.ndarray]] = None
    best_val = -np.inf
    for _ in range(max(n_samples, 1)):
        split = random_split()
        val, _ = score(split)
        if val > best_val:
            best_val, best_split = val, split
    assert best_split is not None

    # coordinate-wise hill-climb polish from the best random start
    cur_split = [np.array(w, dtype=float) for w in best_split]
    cur_val = best_val
    step = 0.25
    for _ in range(max(n_polish, 0)):
        ti = int(rng.integers(0, len(inst.terminals)))
        npaths = n_paths[ti]
        if npaths < 2:
            continue
        i, j = rng.choice(npaths, size=2, replace=False)
        delta = min(step, float(cur_split[ti][i])) * float(rng.random())
        if delta <= 0:
            continue
        trial = [w.copy() for w in cur_split]
        trial[ti][i] -= delta
        trial[ti][j] += delta
        val, _ = score(trial)
        if val > cur_val:
            cur_val, cur_split = val, trial
        else:
            step *= 0.95

    float_best_split = [list(w) for w in cur_split]
    float_best_t = cur_val

    exact_split, exact_t = best_rational_split(inst, float_best_split)

    dmax_exact = inst.d_max()
    rho = float((dmax_exact + exact_t) / dmax_exact)

    return {
        "float_best_t": float_best_t,
        "float_best_split": float_best_split,
        "exact_best_split": exact_split,
        "exact_best_t": exact_t,
        "d_max": dmax_exact,
        "rho": rho,
        "n_routings": len(routings),
        "n_terminals": len(inst.terminals),
        "n_arcs": len(arcs),
    }


# ---------------------------------------------------------------------------
# random_layered_dag: instance generator
# ---------------------------------------------------------------------------

class _TooManyPaths(Exception):
    pass


def random_layered_dag(
    n_layers: int,
    width: int,
    n_terminals: int,
    seed: int,
    demand_pool: Sequence[Sequence[int]],
    max_routings: int = 2500,
    edge_prob: float = 0.5,
    max_attempts: int = 50,
) -> Instance:
    """Random layered DAG: source `s`, `n_layers` internal vertex layers
    (width `width` each, arcs only forward layer L -> L+1 so acyclicity is
    structural, not checked), terminals at the LAST layer only (a
    simplification -- true "last layer(s), plural" spread is future work).

    Every non-source layer's vertices are guaranteed >=1 incoming arc (so no
    terminal is silently unreachable). All s->t paths are enumerated per
    terminal; if the product of per-terminal path counts would exceed
    `max_routings`, the whole graph is resampled with a fresh seed derivation
    (up to `max_attempts` times) rather than silently truncating -- a
    truncated path set would corrupt every downstream verdict.

    Demands: drawn from `demand_pool`, entries of which are themselves full
    demand vectors (e.g. the non-divisible triples (15,10,15), (7,5,3),
    (9,6,4)). If a pool entry of length == n_terminals exists, one is chosen
    uniformly; otherwise individual demands are drawn (with replacement) from
    the flattened pool so any n_terminals is servable.
    """
    assert n_layers >= 1, "need at least one internal layer"
    assert width >= n_terminals >= 1, "width must be >= n_terminals (terminals live in the last layer)"

    for attempt in range(max_attempts):
        rng = random.Random(f"{seed}:{attempt}")
        layers = [[f"L{layer}_{i}" for i in range(width)] for layer in range(n_layers)]
        arcs: List[Arc] = [("s", v) for v in layers[0]]

        for layer in range(n_layers - 1):
            for u in layers[layer]:
                for v in layers[layer + 1]:
                    if rng.random() < edge_prob:
                        arcs.append((u, v))
            incoming = {v: False for v in layers[layer + 1]}
            for (u, v) in arcs:
                if v in incoming:
                    incoming[v] = True
            for v, has_in in incoming.items():
                if not has_in:
                    u = rng.choice(layers[layer])
                    arcs.append((u, v))

        adj: Dict[str, List[str]] = {}
        for (u, v) in arcs:
            adj.setdefault(u, []).append(v)

        last_layer = layers[-1]
        term_names = rng.sample(last_layer, n_terminals)

        def enumerate_paths(target: str, cap: int) -> Optional[List[Tuple[Arc, ...]]]:
            found: List[Tuple[Arc, ...]] = []

            def dfs(node: str, path: List[Arc]) -> None:
                if len(found) > cap:
                    raise _TooManyPaths()
                if node == target:
                    found.append(tuple(path))
                    return
                for nxt in adj.get(node, []):
                    dfs(nxt, path + [(node, nxt)])

            try:
                dfs("s", [])
            except _TooManyPaths:
                return None
            return found

        path_lists = [enumerate_paths(t, cap=max_routings * 2) for t in term_names]
        if any(pl is None or len(pl) == 0 for pl in path_lists):
            continue

        n_routings = 1
        overflow = False
        for pl in path_lists:
            n_routings *= len(pl)
            if n_routings > max_routings:
                overflow = True
                break
        if overflow:
            continue

        matching = [tuple(d) for d in demand_pool if len(d) == n_terminals]
        if matching:
            demands = list(rng.choice(matching))
        else:
            pool_vals = [v for d in demand_pool for v in d]
            demands = [rng.choice(pool_vals) for _ in range(n_terminals)]

        terminals = [
            Terminal(name=term_names[i], demand=Fr(demands[i]), paths=path_lists[i])
            for i in range(n_terminals)
        ]
        used_arcs = sorted({a for t in terminals for path in t.paths for a in path})
        inst = Instance(arcs=used_arcs, terminals=terminals)

        problems = check_paths_valid(inst, "s")
        if problems:
            continue
        return inst

    raise RuntimeError(
        f"random_layered_dag: no valid instance within routing cap after "
        f"{max_attempts} attempts (n_layers={n_layers}, width={width}, "
        f"n_terminals={n_terminals}, max_routings={max_routings})"
    )


# ---------------------------------------------------------------------------
# sweep: run screen_instance over many random instances
# ---------------------------------------------------------------------------

DEFAULT_DEMAND_POOL: Tuple[Tuple[int, ...], ...] = (
    (15, 10, 15), (7, 5, 3), (9, 6, 4),
    (2, 2, 2), (5, 5, 5), (1, 2, 3), (4, 4, 4), (3, 5, 7), (6, 10, 15),
)


def sweep(n_instances: int = 30, seed: int = 0, n_samples: int = 150,
          n_polish: int = 30, demand_pool: Sequence[Sequence[int]] = DEFAULT_DEMAND_POOL
          ) -> Dict:
    """Run screen_instance over `n_instances` random layered DAGs. Reports
    max/min/mean rho = (d_max + t*)/d_max (t* here is the screening lower
    bound on the true sup, so rho is itself a lower-bound estimate) and
    flags -- LOUDLY, to stdout as well as the returned dict -- any instance
    where the exact-rational t came out > 0 (a genuine counterexample
    candidate; none are expected from unstructured random search per the
    prior-session anchor).
    """
    rng = random.Random(seed)
    results: List[Dict] = []
    counterexamples: List[Dict] = []

    structural_choices = [
        (n_layers, width, n_terms)
        for n_layers in (2, 3)
        for width in (2, 3)
        for n_terms in (2, 3)
        if width >= n_terms
    ]

    i = 0
    tries = 0
    while i < n_instances and tries < n_instances * 5:
        tries += 1
        n_layers, width, n_terminals = rng.choice(structural_choices)
        inst_seed = rng.randrange(10**9)
        try:
            inst = random_layered_dag(
                n_layers, width, n_terminals, inst_seed, demand_pool,
            )
        except RuntimeError:
            continue

        res = screen_instance(inst, n_samples=n_samples, n_polish=n_polish, seed=inst_seed)
        res["params"] = {
            "n_layers": n_layers, "width": width, "n_terminals": n_terminals,
            "seed": inst_seed,
        }
        res["instance_repr"] = repr(inst)
        results.append(res)
        if res["exact_best_t"] > 0:
            counterexamples.append(res)
        i += 1

    rhos = [r["rho"] for r in results]
    summary = {
        "n_instances_run": len(results),
        "max_rho": max(rhos) if rhos else None,
        "min_rho": min(rhos) if rhos else None,
        "mean_rho": (sum(rhos) / len(rhos)) if rhos else None,
        "counterexamples_found": len(counterexamples),
        "counterexample_reprs": [c["instance_repr"] for c in counterexamples],
        "results": results,
    }

    if counterexamples:
        print("=" * 70)
        print("!!! POTENTIAL CONJECTURE 1.3 COUNTEREXAMPLE FOUND !!!")
        for c in counterexamples:
            print(f"  exact_best_t = {c['exact_best_t']}  (should be <= 0)")
            print(f"  params = {c['params']}")
            print("  instance repr saved below for exact re-verification:")
            print(f"  {c['instance_repr']}")
        print("=" * 70)

    return summary


if __name__ == "__main__":
    import time

    t0 = time.time()
    summary = sweep(n_instances=30, seed=42)
    elapsed = time.time() - t0
    print(f"ran {summary['n_instances_run']} instances in {elapsed:.1f}s")
    print(f"max rho  = {summary['max_rho']:.4f}")
    print(f"min rho  = {summary['min_rho']:.4f}")
    print(f"mean rho = {summary['mean_rho']:.4f}")
    print(f"counterexamples found: {summary['counterexamples_found']}")
    if summary["counterexamples_found"] == 0:
        print("no counterexamples -- consistent with the plateau-below-1 anchor")
