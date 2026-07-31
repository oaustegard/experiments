"""Arm A — synthetic probe of meet/join asymmetry under a half-space geometry.

Thesis under test (Lattice Representation Hypothesis, arXiv 2603.01227):
if attribute ``m`` holds of object ``g`` iff ``<v_m, e_g> > tau_m``, then the
extents realizable by the geometry are exactly the intersections of the
attribute half-spaces. Intersections of half-spaces are closed under
intersection, so the FCA **meet** extent ``A1 & A2`` is realized exactly.
Unions of half-space intersections are *not* half-space intersections, so the
FCA **join** extent ``(A1 | A2)''`` must over-approximate ``A1 | A2``. The
objects in the difference are "phantoms": admitted by the geometry, in neither
input extent.

This arm quantifies that overshoot on fully synthetic contexts, and asks how it
moves with (2) embedding dimension, (3) attribute-direction coherence at fixed
dimension, and (4) attribute density.

Everything FCA-side is imported from ``fca.py`` (shared, verified core). The
only thing reimplemented here is a *vectorized batch* form of the closure, so
that hundreds of thousands of concept pairs are tractable; it is checked
element-for-element against ``fca.join_overshoot`` / ``fca.meet_extent`` on
random samples in every single configuration (see ``_crosscheck``), and against
a third, arithmetically-disjoint route straight from ``(E, V, tau)`` that uses
per-attribute minima rather than boolean ``all`` (see ``geometric_closure``).

Run:  python3 arm_a_synthetic.py
Writes: ../results/arm_a_synthetic.json
"""

from __future__ import annotations

import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fca  # noqa: E402  (shared verified core)

# --------------------------------------------------------------------------
# Global experiment constants. Every one of these lands in the JSON.
# --------------------------------------------------------------------------

N_OBJ = 200            # objects per synthetic context
N_ATT = 12             # attributes; 2**12 = 4096 is also the hard ceiling on concepts
MAX_CONCEPTS = 5000    # > 2**N_ATT, so for N_ATT=12 this cap can never bind
MAX_PAIRS = 50_000     # concept pairs sampled per (config, seed) when exhaustive is too big
CROSSCHECK_PAIRS = 40  # pairs re-verified against fca.py's scalar path, per config

BASELINE_SEEDS = list(range(12))
SWEEP_SEEDS = list(range(8))

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "arm_a_synthetic.json"


# --------------------------------------------------------------------------
# Context generation
# --------------------------------------------------------------------------


def make_directions(rng: np.random.Generator, n_att: int, d: int, r: float = 0.0) -> np.ndarray:
    """``n_att`` unit directions in R^d with a tunable shared component.

    ``r = 0`` gives iid Gaussian directions (coherence ~ the random baseline for
    that ``d``). ``r -> 1`` collapses every direction onto one shared axis. The
    ``sqrt(1-r) / sqrt(r)`` weighting keeps the pre-normalization variance
    constant so ``r`` reads directly as a correlation, not just "more shared".
    """
    G = rng.normal(size=(n_att, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    if r > 0.0:
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        G = math.sqrt(1.0 - r) * G + math.sqrt(r) * u[None, :]
        G /= np.linalg.norm(G, axis=1, keepdims=True)
    return G


def direction_stats(V: np.ndarray) -> dict:
    """Coherence, but not only the max.

    ``fca.coherence`` is the *max* |cos|, which is the right statistic for the
    paper's linear-independence condition but a poor summary of a whole
    direction set: a low-``d`` random set and a shared-direction set can match
    on the max while differing completely in the bulk, and — critically — in
    *sign*. Positively correlated attributes co-occur (large intents);
    negatively correlated ones exclude each other (small intents). Those pull
    the closure in opposite directions, so both are recorded.
    """
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = Vn @ Vn.T
    iu = np.triu_indices(len(V), k=1)
    c = C[iu]
    return {
        "coherence_max_abs": float(np.abs(c).max()) if c.size else 0.0,
        "cos_mean_abs": float(np.abs(c).mean()) if c.size else 0.0,
        "cos_mean_signed": float(c.mean()) if c.size else 0.0,
        "cos_min_signed": float(c.min()) if c.size else 0.0,
        "cos_frac_negative": float((c < 0).mean()) if c.size else 0.0,
    }


def build(seed: int, d: int, n_att: int, p: float, r: float = 0.0, n_obj: int = N_OBJ):
    """Random embeddings + directions + per-attribute quantile thresholds.

    ``tau_m`` is set to the ``(1-p)`` quantile of the projections onto ``v_m``,
    so *every* attribute has empirical density ~= ``p`` by construction. That
    decouples the density knob from ``d`` and ``r``: without it, correlated
    directions would also drift the densities and confound sweeps (3) and (4)
    with sweep (2).
    """
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_obj, d))
    V = make_directions(rng, n_att, d, r)
    S = E @ V.T
    tau = np.quantile(S, 1.0 - p, axis=0)
    ctx = fca.realize(E, V, tau)
    return E, V, tau, S, ctx


# --------------------------------------------------------------------------
# Vectorized closure over many pairs at once
# --------------------------------------------------------------------------


def _derive_batch(notI: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched ``(A', A'')``. ``A`` is (P, n_obj) bool; returns (P, n_att), (P, n_obj).

    ``notI`` is ``(~I)`` as float32. ``A @ notI`` counts, for each attribute,
    how many objects of ``A`` *lack* it; zero means "shared by all of A", i.e.
    ``A'``. The second matmul counts, for each object, how many attributes of
    ``A'`` it lacks; zero means "carries all of them", i.e. ``A''``. Empty
    inputs give all-zero counts, which reproduces fca.py's vacuous-truth
    convention (empty object set derives all of M, empty attribute set derives
    all of G) without a special case.
    """
    B = (A.astype(np.float32) @ notI) == 0.0          # (P, n_att)  = A'
    return B, (B.astype(np.float32) @ notI.T) == 0.0  # (P, n_obj)  = A''


def _close_batch(notI: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Batched ``A''`` only."""
    return _derive_batch(notI, A)[1]


def geometric_closure(S: np.ndarray, tau: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Closure computed straight from the geometry, by a disjoint arithmetic route.

    ``A'`` = the attributes whose half-space contains all of ``A`` = those with
    ``min_{g in A} <v_m, e_g> > tau_m``. ``A''`` = the objects inside the
    intersection of exactly those half-spaces. Uses ``min`` over raw
    projections instead of boolean ``all`` over the incidence, so it shares no
    intermediate with either ``fca.close_objs`` or ``_close_batch``. This is
    also the direct statement that the FCA closure *is* the half-space
    intersection under this model.
    """
    if not A.any():
        keep = np.ones(S.shape[1], bool)
    else:
        keep = S[A].min(axis=0) > tau
    if not keep.any():
        return np.ones(S.shape[0], bool)
    return np.all(S[:, keep] > tau[keep], axis=1)


def pair_metrics(ctx: fca.Context, extents: np.ndarray, pairs: np.ndarray, chunk: int = 4096) -> dict:
    """Meet/join overshoot for a batch of concept-index pairs.

    Returns per-pair arrays plus the comparability mask. A pair with one extent
    contained in the other has ``A1 | A2`` already closed, hence zero overshoot
    for free; those pairs are reported separately so they cannot flatter the
    headline number.
    """
    notI = (~ctx.I).astype(np.float32)
    n = len(pairs)
    join_over = np.empty(n, np.float64)
    join_rate = np.empty(n, np.float64)
    join_phantom = np.empty(n, np.int64)
    join_union = np.empty(n, np.int64)
    join_closed = np.empty(n, np.int64)
    join_intent = np.empty(n, np.int64)
    meet_phantom = np.empty(n, np.int64)
    comparable = np.empty(n, bool)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        A1 = extents[pairs[lo:hi, 0]]
        A2 = extents[pairs[lo:hi, 1]]

        union = A1 | A2
        # B1 & B2 — the half-spaces that survive to define the join. This is the
        # proposed mediator: the closure is the intersection of exactly these,
        # so more survivors means a tighter closure means less overshoot.
        surv, closed = _derive_batch(notI, union)
        nu = union.sum(1)
        nc = closed.sum(1)
        ph = (closed & ~union).sum(1)
        join_intent[lo:hi] = surv.sum(1)
        join_union[lo:hi] = nu
        join_closed[lo:hi] = nc
        join_phantom[lo:hi] = ph
        join_over[lo:hi] = np.where(nc > 0, ph / np.maximum(nc, 1), 0.0)
        # Exposure normalization: of the objects OUTSIDE the union — the only
        # ones that could possibly be admitted as phantoms — what fraction is?
        # phantoms/|closed| is not comparable across configurations whose extents
        # differ in size, because the denominator inflates with the union. This
        # one holds exposure fixed.
        elig = ctx.n_obj - nu
        join_rate[lo:hi] = np.where(elig > 0, ph / np.maximum(elig, 1), 0.0)

        inter = A1 & A2
        meet_phantom[lo:hi] = (_close_batch(notI, inter) & ~inter).sum(1)

        comparable[lo:hi] = ((A1 & ~A2).sum(1) == 0) | ((A2 & ~A1).sum(1) == 0)

    return {
        "join_overshoot": join_over,
        "join_phantom_rate": join_rate,
        "join_phantoms": join_phantom,
        "join_union": join_union,
        "join_closed": join_closed,
        "join_intent": join_intent,
        "meet_phantoms": meet_phantom,
        "comparable": comparable,
    }


def _crosscheck(ctx: fca.Context, S, tau, extents, pairs, rng) -> None:
    """Re-verify the vectorized path against fca.py and against the geometry.

    METHODS.md principle 1: the check must not share the modelling assumption.
    Route 1 is ``_close_batch`` (boolean matmul). Route 2 is
    ``fca.join_overshoot`` (numpy ``all`` over fancy-indexed rows, written by
    someone else). Route 3 is ``geometric_closure`` (per-attribute minima over
    the raw projections). All three must agree exactly.
    """
    notI = (~ctx.I).astype(np.float32)
    idx = rng.choice(len(pairs), size=min(CROSSCHECK_PAIRS, len(pairs)), replace=False)
    for k in idx:
        i, j = pairs[k]
        A1, A2 = extents[i], extents[j]
        ref = fca.join_overshoot(ctx, A1, A2)
        mine = _close_batch(notI, (A1 | A2)[None, :])[0]
        geo = geometric_closure(S, tau, A1 | A2)
        assert int(mine.sum()) == ref["closed"], "batch closure != fca.join_overshoot"
        assert np.array_equal(mine, geo), "batch closure != geometric closure"
        assert np.array_equal(fca.join_extent(ctx, A1, A2), mine), "join_extent mismatch"
        m = fca.meet_extent(ctx, A1, A2)
        assert np.array_equal(_close_batch(notI, m[None, :])[0], m), "meet not closed"


# --------------------------------------------------------------------------
# One (config, seed) run
# --------------------------------------------------------------------------


HIST_EDGES = [0.0, 1e-12, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0001]
HIST_LABELS = ["exactly 0", "(0,0.02]", "(0.02,0.05]", "(0.05,0.10]", "(0.10,0.20]",
               "(0.20,0.30]", "(0.30,0.50]", "(0.50,0.75]", "(0.75,1.0]"]


def histogram(x: np.ndarray) -> list[int]:
    """Counts in HIST_EDGES bins. Bin 0 is *exactly* zero, kept separate on purpose:
    the share of joins the geometry gets right for free is a headline number and
    must not be smeared into a near-zero bucket."""
    if x.size == 0:
        return [0] * len(HIST_LABELS)
    h = [int((x == 0.0).sum())]
    for lo, hi in zip(HIST_EDGES[1:-1], HIST_EDGES[2:]):
        h.append(int(((x > lo) & (x <= hi)).sum()))
    return h


def summarize(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "frac_zero": None}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
        "frac_zero": float(np.mean(x == 0.0)),
    }


def run_one(seed: int, d: int, n_att: int, p: float, r: float = 0.0, n_obj: int = N_OBJ) -> dict:
    E, V, tau, S, ctx = build(seed, d, n_att, p, r, n_obj)
    concepts = ctx.concepts(max_concepts=MAX_CONCEPTS)
    n_con = len(concepts)
    cap_hit = n_con >= MAX_CONCEPTS
    extents = np.array([c[0] for c in concepts], dtype=bool)

    n_pairs_total = n_con * (n_con - 1) // 2
    prng = np.random.default_rng(10_000 + seed)
    if n_pairs_total == 0:
        pairs = np.zeros((0, 2), dtype=np.int64)
        exhaustive = True
    elif n_pairs_total <= MAX_PAIRS:
        pairs = np.array(list(itertools.combinations(range(n_con), 2)), dtype=np.int64)
        exhaustive = True
    else:
        # sample distinct unordered pairs without materializing all of them
        flat = prng.choice(n_pairs_total, size=MAX_PAIRS, replace=False)
        # invert the linear index of the strict upper triangle
        ii = (n_con - 2 - np.floor(np.sqrt(-8 * flat + 4 * n_con * (n_con - 1) - 7) / 2.0 - 0.5)).astype(np.int64)
        jj = (flat + ii + 1 - n_con * (n_con - 1) // 2 + (n_con - ii) * ((n_con - ii) - 1) // 2).astype(np.int64)
        pairs = np.stack([ii, jj], axis=1)
        assert (pairs[:, 0] < pairs[:, 1]).all() and pairs.max() < n_con
        assert len(np.unique(pairs, axis=0)) == len(pairs), "pair sampling produced duplicates"
        exhaustive = False

    m = pair_metrics(ctx, extents, pairs)
    if len(pairs):
        _crosscheck(ctx, S, tau, extents, pairs, prng)

    inc = ~m["comparable"]
    over = m["join_overshoot"]
    intents = np.array([c[1] for c in concepts], dtype=bool)

    out = {
        "seed": seed,
        "d": d,
        "n_att": n_att,
        "p_target": p,
        "r": r,
        "n_obj": n_obj,
        "density_actual": float(ctx.I.mean()),
        "coherence": float(fca.coherence(V)),
        "rank_V": int(np.linalg.matrix_rank(V)),
        "n_concepts": n_con,
        "concept_cap": MAX_CONCEPTS,
        "concept_cap_hit": bool(cap_hit),
        "realizable_intent_fraction": n_con / float(2 ** n_att),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_used": int(len(pairs)),
        "pairs_exhaustive": bool(exhaustive),
        "frac_pairs_incomparable": float(inc.mean()) if len(pairs) else None,
        "join_overshoot_all_pairs": summarize(over),
        "join_overshoot_incomparable_pairs": summarize(over[inc]),
        "phantom_rate_all_pairs": summarize(m["join_phantom_rate"]),
        "phantom_rate_incomparable_pairs": summarize(m["join_phantom_rate"][inc]),
        "join_phantoms_all_pairs_mean": float(m["join_phantoms"].mean()) if len(pairs) else None,
        "join_phantoms_incomparable_mean": float(m["join_phantoms"][inc].mean()) if inc.any() else None,
        "mean_union_size": float(m["join_union"].mean()) if len(pairs) else None,
        "mean_closed_size": float(m["join_closed"].mean()) if len(pairs) else None,
        "meet_phantoms_max": int(m["meet_phantoms"].max()) if len(pairs) else 0,
        "meet_phantoms_total": int(m["meet_phantoms"].sum()) if len(pairs) else 0,
        "mean_extent_size": float(extents.sum(1).mean()) if n_con else None,
        "mean_intent_size": float(intents.sum(1).mean()) if n_con else None,
        "mean_join_intent_size": float(m["join_intent"].mean()) if len(pairs) else None,
        "mean_join_intent_size_incomparable": float(m["join_intent"][inc].mean()) if inc.any() else None,
        "overshoot_hist_all": histogram(over),
        "overshoot_hist_incomparable": histogram(over[inc]),
    }
    out.update(direction_stats(V))
    return out


# --------------------------------------------------------------------------
# Aggregation across seeds
# --------------------------------------------------------------------------


def agg(runs: list[dict], path: str) -> dict:
    """mean / std / min / max of one dotted field across seed-runs."""
    vals = []
    for rr in runs:
        cur = rr
        for k in path.split("."):
            cur = cur[k]
        if cur is not None:
            vals.append(float(cur))
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n_seeds": 0}
    a = np.array(vals)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n_seeds": len(a),
    }


AGG_FIELDS = [
    "coherence",
    "cos_mean_abs",
    "cos_mean_signed",
    "cos_min_signed",
    "cos_frac_negative",
    "rank_V",
    "n_concepts",
    "realizable_intent_fraction",
    "density_actual",
    "frac_pairs_incomparable",
    "join_overshoot_all_pairs.mean",
    "join_overshoot_all_pairs.median",
    "join_overshoot_all_pairs.p90",
    "join_overshoot_all_pairs.max",
    "join_overshoot_all_pairs.frac_zero",
    "join_overshoot_incomparable_pairs.mean",
    "join_overshoot_incomparable_pairs.median",
    "join_overshoot_incomparable_pairs.p90",
    "join_overshoot_incomparable_pairs.max",
    "join_overshoot_incomparable_pairs.frac_zero",
    "phantom_rate_all_pairs.mean",
    "phantom_rate_all_pairs.median",
    "phantom_rate_incomparable_pairs.mean",
    "phantom_rate_incomparable_pairs.median",
    "phantom_rate_incomparable_pairs.p90",
    "join_phantoms_all_pairs_mean",
    "join_phantoms_incomparable_mean",
    "mean_union_size",
    "mean_closed_size",
    "mean_extent_size",
    "mean_intent_size",
    "mean_join_intent_size",
    "mean_join_intent_size_incomparable",
    "meet_phantoms_max",
    "meet_phantoms_total",
]


def cell(label: dict, runs: list[dict]) -> dict:
    out = dict(label)
    out["n_seeds"] = len(runs)
    out["any_concept_cap_hit"] = any(r["concept_cap_hit"] for r in runs)
    out["all_pairs_exhaustive"] = all(r["pairs_exhaustive"] for r in runs)
    out["pairs_used_total"] = int(sum(r["n_pairs_used"] for r in runs))
    out["agg"] = {f: agg(runs, f) for f in AGG_FIELDS}
    out["hist_labels"] = HIST_LABELS
    out["pooled_overshoot_hist_all"] = [
        int(sum(r["overshoot_hist_all"][b] for r in runs)) for b in range(len(HIST_LABELS))
    ]
    out["pooled_overshoot_hist_incomparable"] = [
        int(sum(r["overshoot_hist_incomparable"][b] for r in runs)) for b in range(len(HIST_LABELS))
    ]
    out["per_seed"] = runs
    return out


# --------------------------------------------------------------------------
# Calibration gates — fca.py's own, run unmodified
# --------------------------------------------------------------------------


def run_gates() -> dict:
    """Both fca.py gates, on two independent contexts.

    ``assert_meet_closed`` is the known-good side (a theorem — a raise means the
    core is broken, not that we found something). ``assert_join_overshoots`` is
    the known-bad side (a raise means the context is degenerate and the study is
    vacuous). Both are O(n_concepts^2) Python loops, so they run on a small
    context and on a truncated prefix of a full-size one; the truncation is
    recorded rather than hidden.
    """
    gates = []

    # (a) small context, exhaustive
    _, _, _, _, ctx = build(seed=0, d=32, n_att=7, p=0.5, n_obj=60)
    cs = ctx.concepts(max_concepts=MAX_CONCEPTS)
    t0 = time.time()
    n_meet = fca.assert_meet_closed(ctx, cs)
    n_join = fca.assert_join_overshoots(ctx, cs)
    gates.append({
        "context": "small: d=32 n_att=7 p=0.5 n_obj=60 seed=0",
        "n_concepts": len(cs),
        "concepts_truncated_for_gate": False,
        "assert_meet_closed_pairs_checked": int(n_meet),
        "assert_meet_closed_passed": True,
        "assert_join_overshoots_overshooting_pairs": int(n_join),
        "assert_join_overshoots_passed": True,
        "seconds": round(time.time() - t0, 3),
    })

    # (b) full-size baseline context, first 250 concepts only (O(N^2) Python)
    _, _, _, _, ctx2 = build(seed=0, d=64, n_att=N_ATT, p=0.5, n_obj=N_OBJ)
    cs2_full = ctx2.concepts(max_concepts=MAX_CONCEPTS)
    trunc = 250
    cs2 = cs2_full[:trunc]
    t0 = time.time()
    n_meet2 = fca.assert_meet_closed(ctx2, cs2)
    n_join2 = fca.assert_join_overshoots(ctx2, cs2)
    gates.append({
        "context": f"baseline: d=64 n_att={N_ATT} p=0.5 n_obj={N_OBJ} seed=0",
        "n_concepts": len(cs2_full),
        "concepts_truncated_for_gate": True,
        "gate_concept_prefix": trunc,
        "assert_meet_closed_pairs_checked": int(n_meet2),
        "assert_meet_closed_passed": True,
        "assert_join_overshoots_overshooting_pairs": int(n_join2),
        "assert_join_overshoots_passed": True,
        "seconds": round(time.time() - t0, 3),
    })
    return {"gates": gates, "both_gates_passed": True}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    results: dict = {
        "arm": "A — synthetic half-space realization",
        "thesis": (
            "FCA meet extent A1&A2 is a bare intersection and is realized exactly by "
            "half-space intersection; FCA join extent (A1|A2)'' is a closure of a union, "
            "which half-spaces cannot represent, so the geometry must over-approximate "
            "and admit phantom objects."
        ),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpus": os.cpu_count(),
        },
        "global_config": {
            "n_obj": N_OBJ,
            "n_att_default": N_ATT,
            "max_concepts_cap": MAX_CONCEPTS,
            "max_pairs_per_seed": MAX_PAIRS,
            "crosscheck_pairs_per_config": CROSSCHECK_PAIRS,
            "baseline_seeds": BASELINE_SEEDS,
            "sweep_seeds": SWEEP_SEEDS,
            "threshold_rule": "tau_m = quantile(E @ v_m, 1-p) so every attribute has density ~= p",
            "embedding_dist": "iid standard normal",
            "notes": [
                "MAX_CONCEPTS=5000 exceeds 2**12=4096, the hard ceiling on intents for "
                "n_att=12, so the concept cap can never bind in any configuration run here.",
                "Pair sampling is uniform without replacement over the strict upper "
                "triangle when C(n_concepts,2) > MAX_PAIRS; otherwise all pairs are used.",
                "Overshoot is reported both over all concept pairs and over incomparable "
                "pairs only. Comparable pairs (A1 subset A2) have a union that is already "
                "closed, so they contribute a structural zero and would flatter the mean.",
            ],
        },
    }

    print("== calibration gates ==")
    results["calibration"] = run_gates()
    for g in results["calibration"]["gates"]:
        print(f"  {g['context']}: meet-closed OK over {g['assert_meet_closed_pairs_checked']} pairs; "
              f"join overshoots on {g['assert_join_overshoots_overshooting_pairs']} pairs "
              f"({g['seconds']}s)")

    # ---- (1) baseline ----------------------------------------------------
    print("\n== (1) baseline: d=64, n_att=12, p=0.5 ==")
    base_runs = [run_one(s, d=64, n_att=N_ATT, p=0.5) for s in BASELINE_SEEDS]
    results["experiment_1_baseline"] = cell({"d": 64, "n_att": N_ATT, "p": 0.5, "r": 0.0}, base_runs)
    a = results["experiment_1_baseline"]["agg"]
    print(f"  concepts {a['n_concepts']['mean']:.0f}+-{a['n_concepts']['std']:.0f}  "
          f"coh {a['coherence']['mean']:.3f}")
    print(f"  join overshoot (all pairs)  mean {a['join_overshoot_all_pairs.mean']['mean']:.4f} "
          f"median {a['join_overshoot_all_pairs.median']['mean']:.4f} "
          f"p90 {a['join_overshoot_all_pairs.p90']['mean']:.4f} "
          f"max {a['join_overshoot_all_pairs.max']['mean']:.4f} "
          f"frac_zero {a['join_overshoot_all_pairs.frac_zero']['mean']:.4f}")
    print(f"  join overshoot (incomparable) mean {a['join_overshoot_incomparable_pairs.mean']['mean']:.4f} "
          f"frac_zero {a['join_overshoot_incomparable_pairs.frac_zero']['mean']:.4f}")
    print(f"  phantom RATE (incomparable, phantoms/outside-objects) mean "
          f"{a['phantom_rate_incomparable_pairs.mean']['mean']:.4f} "
          f"+-{a['phantom_rate_incomparable_pairs.mean']['std']:.4f}")
    print(f"  meet phantoms total across all seeds: {a['meet_phantoms_total']['max']:.0f} (must be 0)")
    print(f"  pooled overshoot histogram (all pairs):")
    for lab, cnt in zip(HIST_LABELS, results["experiment_1_baseline"]["pooled_overshoot_hist_all"]):
        print(f"    {lab:>13}: {cnt}")

    # ---- (2) dimension sweep --------------------------------------------
    print("\n== (2) dimension sweep (n_att=12, p=0.5, r=0) ==")
    dims = [64, 32, 16, 12, 8, 6, 4, 3, 2]
    dim_cells = []
    for d in dims:
        runs = [run_one(s, d=d, n_att=N_ATT, p=0.5) for s in SWEEP_SEEDS]
        c = cell({"d": d, "n_att": N_ATT, "p": 0.5, "r": 0.0}, runs)
        dim_cells.append(c)
        g = c["agg"]
        print(f"  d={d:>2}  cohmax {g['coherence']['mean']:.3f}  |cos| {g['cos_mean_abs']['mean']:.3f}  "
              f"cos {g['cos_mean_signed']['mean']:+.3f}  rankV {g['rank_V']['mean']:.1f}  "
              f"concepts {g['n_concepts']['mean']:7.1f}  int/2^m {g['realizable_intent_fraction']['mean']:.4f}  "
              f"|B1&B2| {g['mean_join_intent_size_incomparable']['mean']:.2f}  "
              f"phant {g['join_phantoms_incomparable_mean']['mean']:5.1f}  "
              f"over_inc {g['join_overshoot_incomparable_pairs.mean']['mean']:.4f}  "
              f"RATE {g['phantom_rate_incomparable_pairs.mean']['mean']:.4f}  "
              f"incomp {g['frac_pairs_incomparable']['mean']:.3f}")
    results["experiment_2_dimension_sweep"] = {
        "swept": "d",
        "values": dims,
        "fixed": {"n_att": N_ATT, "p": 0.5, "r": 0.0, "n_obj": N_OBJ},
        "note": ("d < n_att makes linearly independent attribute directions impossible; "
                 "rank_V is recorded to show exactly where that binds."),
        "cells": dim_cells,
    }

    # ---- (3) coherence sweep at fixed d ---------------------------------
    print("\n== (3) coherence sweep (d=64, n_att=12, p=0.5) ==")
    rs = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    coh_cells = []
    for r in rs:
        runs = [run_one(s, d=64, n_att=N_ATT, p=0.5, r=r) for s in SWEEP_SEEDS]
        c = cell({"d": 64, "n_att": N_ATT, "p": 0.5, "r": r}, runs)
        coh_cells.append(c)
        g = c["agg"]
        print(f"  r={r:.2f}  cohmax {g['coherence']['mean']:.3f}  |cos| {g['cos_mean_abs']['mean']:.3f}  "
              f"cos {g['cos_mean_signed']['mean']:+.3f}  concepts {g['n_concepts']['mean']:7.1f}  "
              f"|B1&B2| {g['mean_join_intent_size_incomparable']['mean']:.2f}  "
              f"phant {g['join_phantoms_incomparable_mean']['mean']:5.1f}  "
              f"over_inc {g['join_overshoot_incomparable_pairs.mean']['mean']:.4f}  "
              f"RATE {g['phantom_rate_incomparable_pairs.mean']['mean']:.4f}  "
              f"incomp {g['frac_pairs_incomparable']['mean']:.3f}")
    results["experiment_3_coherence_sweep"] = {
        "swept": "r (shared-direction weight)",
        "values": rs,
        "fixed": {"d": 64, "n_att": N_ATT, "p": 0.5, "n_obj": N_OBJ},
        "note": ("d=64 >> n_att=12, so rank_V stays full throughout: this isolates "
                 "'directions happen to be correlated' from 'not enough dimensions'."),
        "cells": coh_cells,
    }

    # ---- (4) density sweep ----------------------------------------------
    print("\n== (4) density sweep (d=64, n_att=12, r=0) ==")
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    den_cells = []
    for p in ps:
        runs = [run_one(s, d=64, n_att=N_ATT, p=p) for s in SWEEP_SEEDS]
        c = cell({"d": 64, "n_att": N_ATT, "p": p, "r": 0.0}, runs)
        den_cells.append(c)
        g = c["agg"]
        print(f"  p={p:.1f}  concepts {g['n_concepts']['mean']:7.1f}  "
              f"ext {g['mean_extent_size']['mean']:6.2f}  int {g['mean_intent_size']['mean']:5.2f}  "
              f"|B1&B2| {g['mean_join_intent_size_incomparable']['mean']:.2f}  "
              f"phant {g['join_phantoms_incomparable_mean']['mean']:6.1f}  "
              f"over_inc {g['join_overshoot_incomparable_pairs.mean']['mean']:.4f}  "
              f"RATE {g['phantom_rate_incomparable_pairs.mean']['mean']:.4f}  "
              f"incomp {g['frac_pairs_incomparable']['mean']:.3f}")
    results["experiment_4_density_sweep"] = {
        "swept": "p (attribute density)",
        "values": ps,
        "fixed": {"d": 64, "n_att": N_ATT, "r": 0.0, "n_obj": N_OBJ},
        "cells": den_cells,
    }

    # ---- mechanism: mediation by surviving-half-space count --------------
    # Every sweep moves overshoot. The claim is that they all move it through
    # ONE channel: |B1 & B2|, the number of half-spaces that still contain the
    # whole union and therefore still constrain the closure. Pool every cell
    # from every sweep and check whether that single predictor tracks overshoot
    # across all of them. NOTE: this is a post-hoc descriptive fit on the same
    # runs that motivated it — it is not held out, and is reported as a
    # consistency check on a mechanism, not as a validated model.
    all_cells = ([results["experiment_1_baseline"]] + dim_cells + coh_cells + den_cells)
    med_x, med_y, med_lab = [], [], []
    for c in all_cells:
        x = c["agg"]["mean_join_intent_size_incomparable"]["mean"]
        y = c["agg"]["join_overshoot_incomparable_pairs.mean"]["mean"]
        if x is not None and y is not None:
            med_x.append(x)
            med_y.append(y)
            med_lab.append({k: c[k] for k in ("d", "n_att", "p", "r") if k in c})
    mx, my = np.array(med_x), np.array(med_y)

    def _pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    mechanism = {
        "predictor": "mean |B1 & B2| over incomparable concept pairs "
                     "(number of half-spaces still containing the union)",
        "response": "mean join overshoot over incomparable pairs",
        "n_cells_pooled": len(mx),
        "pearson_r": _pearson(mx, my),
        "spearman_rho": _spearman(mx, my),
        "pearson_r_log_predictor": _pearson(np.log(mx + 1e-9), my),
        "cells": [{"config": l, "mean_join_intent_size": float(a), "mean_overshoot": float(b)}
                  for l, a, b in zip(med_lab, mx, my)],
        "caveat": "post-hoc descriptive fit on the same runs; not held out.",
    }

    # matched-coherence dissociation: does max-coherence alone predict overshoot?
    dissociation = []
    for arm, cells_ in (("dimension", dim_cells), ("correlation", coh_cells)):
        for c in cells_:
            g = c["agg"]
            dissociation.append({
                "config": {k: c[k] for k in ("d", "r")},
                "arm": arm,
                "coherence_max_abs": g["coherence"]["mean"],
                "cos_mean_abs": g["cos_mean_abs"]["mean"],
                "cos_mean_signed": g["cos_mean_signed"]["mean"],
                "cos_frac_negative": g["cos_frac_negative"]["mean"],
                "mean_join_intent_size": g["mean_join_intent_size_incomparable"]["mean"],
                "overshoot_incomparable": g["join_overshoot_incomparable_pairs.mean"]["mean"],
            })
    coh_all = np.array([x["coherence_max_abs"] for x in dissociation])
    ovr_all = np.array([x["overshoot_incomparable"] for x in dissociation])
    sgn_all = np.array([x["cos_mean_signed"] for x in dissociation])
    # The headline dissociation: does the *ratio* measure and the *rate* measure
    # agree on the sign of the dimension effect? They agree on the other two
    # sweeps. If they disagree here, "overshoot falls as d falls" is an artifact
    # of the denominator growing, not the geometry improving.
    def _monotone(cells_, key, xs):
        ys = [c["agg"][key]["mean"] for c in cells_]
        rho = _spearman(np.array(xs, float), np.array(ys, float))
        return {"x": list(xs), "y": ys, "spearman_rho_vs_x": rho,
                "direction": "increasing" if rho > 0.5 else ("decreasing" if rho < -0.5 else "non-monotonic")}

    mechanism["normalization_dissociation"] = {
        "why": ("phantoms/|closed| divides by a denominator that itself grows when "
                "extents inflate; phantoms/|objects outside the union| holds the "
                "number of objects at risk fixed. Where the two disagree in sign, "
                "the ratio measure is the misleading one."),
        "dimension_sweep_vs_d": {
            "ratio_phantoms_over_closed": _monotone(
                dim_cells, "join_overshoot_incomparable_pairs.mean", dims),
            "rate_phantoms_over_outside": _monotone(
                dim_cells, "phantom_rate_incomparable_pairs.mean", dims),
            "absolute_phantom_count": _monotone(
                dim_cells, "join_phantoms_incomparable_mean", dims),
            "mean_union_size": _monotone(dim_cells, "mean_union_size", dims),
        },
        "coherence_sweep_vs_r": {
            "ratio_phantoms_over_closed": _monotone(
                coh_cells, "join_overshoot_incomparable_pairs.mean", rs),
            "rate_phantoms_over_outside": _monotone(
                coh_cells, "phantom_rate_incomparable_pairs.mean", rs),
            "absolute_phantom_count": _monotone(
                coh_cells, "join_phantoms_incomparable_mean", rs),
        },
        "density_sweep_vs_p": {
            "ratio_phantoms_over_closed": _monotone(
                den_cells, "join_overshoot_incomparable_pairs.mean", ps),
            "rate_phantoms_over_outside": _monotone(
                den_cells, "phantom_rate_incomparable_pairs.mean", ps),
            "absolute_phantom_count": _monotone(
                den_cells, "join_phantoms_incomparable_mean", ps),
        },
    }

    mechanism["max_coherence_alone"] = {
        "pearson_r_vs_overshoot_pooling_both_sweeps": _pearson(coh_all, ovr_all),
        "pearson_r_signed_mean_cos_vs_overshoot": _pearson(sgn_all, ovr_all),
        "note": ("If max coherence were the sufficient statistic, the dimension arm and "
                 "the correlation arm would lie on one curve. Compare cells with matched "
                 "coherence_max_abs across arms in `matched_coherence_cells`."),
    }
    mechanism["matched_coherence_cells"] = dissociation
    results["mechanism"] = mechanism

    # ---- derived summary -------------------------------------------------
    def series(cells, key):
        return [c["agg"][key]["mean"] for c in cells]

    results["summary"] = {
        "meet_exactness": {
            "claim": "meet is realized exactly — zero phantoms, always",
            "total_meet_phantoms_over_every_pair_in_every_config": int(sum(
                r["meet_phantoms_total"]
                for blk in ("experiment_1_baseline",)
                for r in results[blk]["per_seed"]
            ) + sum(
                r["meet_phantoms_total"]
                for blk in ("experiment_2_dimension_sweep", "experiment_3_coherence_sweep",
                            "experiment_4_density_sweep")
                for c in results[blk]["cells"]
                for r in c["per_seed"]
            )),
            "total_pairs_checked": int(sum(
                r["n_pairs_used"] for r in results["experiment_1_baseline"]["per_seed"]
            ) + sum(
                r["n_pairs_used"]
                for blk in ("experiment_2_dimension_sweep", "experiment_3_coherence_sweep",
                            "experiment_4_density_sweep")
                for c in results[blk]["cells"]
                for r in c["per_seed"]
            )),
        },
        "dimension_sweep_series": {
            "d": dims,
            "coherence": series(dim_cells, "coherence"),
            "rank_V": series(dim_cells, "rank_V"),
            "n_concepts": series(dim_cells, "n_concepts"),
            "realizable_intent_fraction": series(dim_cells, "realizable_intent_fraction"),
            "join_overshoot_mean_all": series(dim_cells, "join_overshoot_all_pairs.mean"),
            "join_overshoot_median_all": series(dim_cells, "join_overshoot_all_pairs.median"),
            "join_overshoot_mean_incomparable": series(dim_cells, "join_overshoot_incomparable_pairs.mean"),
            "join_overshoot_median_incomparable": series(dim_cells, "join_overshoot_incomparable_pairs.median"),
            "phantom_rate_mean_all": series(dim_cells, "phantom_rate_all_pairs.mean"),
            "phantom_rate_mean_incomparable": series(dim_cells, "phantom_rate_incomparable_pairs.mean"),
            "mean_union_size": series(dim_cells, "mean_union_size"),
            "mean_closed_size": series(dim_cells, "mean_closed_size"),
            "mean_extent_size": series(dim_cells, "mean_extent_size"),
            "mean_join_intent_size_incomparable": series(dim_cells, "mean_join_intent_size_incomparable"),
            "cos_mean_abs": series(dim_cells, "cos_mean_abs"),
            "cos_mean_signed": series(dim_cells, "cos_mean_signed"),
            "frac_zero_all": series(dim_cells, "join_overshoot_all_pairs.frac_zero"),
            "frac_pairs_incomparable": series(dim_cells, "frac_pairs_incomparable"),
            "mean_phantoms_incomparable": series(dim_cells, "join_phantoms_incomparable_mean"),
        },
        "coherence_sweep_series": {
            "r": rs,
            "coherence": series(coh_cells, "coherence"),
            "n_concepts": series(coh_cells, "n_concepts"),
            "join_overshoot_mean_all": series(coh_cells, "join_overshoot_all_pairs.mean"),
            "join_overshoot_mean_incomparable": series(coh_cells, "join_overshoot_incomparable_pairs.mean"),
            "phantom_rate_mean_all": series(coh_cells, "phantom_rate_all_pairs.mean"),
            "phantom_rate_mean_incomparable": series(coh_cells, "phantom_rate_incomparable_pairs.mean"),
            "cos_mean_abs": series(coh_cells, "cos_mean_abs"),
            "cos_mean_signed": series(coh_cells, "cos_mean_signed"),
            "mean_union_size": series(coh_cells, "mean_union_size"),
            "mean_closed_size": series(coh_cells, "mean_closed_size"),
            "mean_extent_size": series(coh_cells, "mean_extent_size"),
            "mean_join_intent_size_incomparable": series(coh_cells, "mean_join_intent_size_incomparable"),
            "frac_pairs_incomparable": series(coh_cells, "frac_pairs_incomparable"),
            "mean_phantoms_incomparable": series(coh_cells, "join_phantoms_incomparable_mean"),
        },
        "density_sweep_series": {
            "p": ps,
            "n_concepts": series(den_cells, "n_concepts"),
            "mean_extent_size": series(den_cells, "mean_extent_size"),
            "join_overshoot_mean_all": series(den_cells, "join_overshoot_all_pairs.mean"),
            "join_overshoot_mean_incomparable": series(den_cells, "join_overshoot_incomparable_pairs.mean"),
            "phantom_rate_mean_all": series(den_cells, "phantom_rate_all_pairs.mean"),
            "phantom_rate_mean_incomparable": series(den_cells, "phantom_rate_incomparable_pairs.mean"),
            "mean_union_size": series(den_cells, "mean_union_size"),
            "mean_closed_size": series(den_cells, "mean_closed_size"),
            "mean_intent_size": series(den_cells, "mean_intent_size"),
            "mean_join_intent_size_incomparable": series(den_cells, "mean_join_intent_size_incomparable"),
            "frac_pairs_incomparable": series(den_cells, "frac_pairs_incomparable"),
            "mean_phantoms_incomparable": series(den_cells, "join_phantoms_incomparable_mean"),
        },
    }

    results["runtime_seconds"] = round(time.time() - t_start, 1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(RESULTS_PATH)
    print(f"\nwrote {RESULTS_PATH} ({RESULTS_PATH.stat().st_size} bytes) in {results['runtime_seconds']}s")


if __name__ == "__main__":
    main()
