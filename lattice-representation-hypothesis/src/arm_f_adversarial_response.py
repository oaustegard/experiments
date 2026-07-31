"""Arm F — checking the adversarial review's own claims.

An adversarial pass over Arms A/D/E defeated most of the original thesis.
Before accepting that, its load-bearing claims get the same treatment
everything else here got: an independent implementation.

Four checks:

  F1  THE CONE IDENTITY. The review claims the paper's "conic hull" clause
      makes Definition 7's join exactly R(Y_A & Y_B) whenever the attribute
      directions involved are linearly independent — which holds throughout
      the paper's own experiments (|M| <= 184, d >= 3072). If true, the
      original "Definition 7 is not a lattice element" complaint collapses
      to an exposition note. Verified here by Minkowski-sum membership LP,
      written from the dual-cone definition rather than reusing the
      review's construction.

  F2  IS A WORDNET-SHAPED CONTEXT BENIGN? The review argues WordNet
      sub-hierarchies are "strongly nested, near a chain", where the closure
      gap vanishes. A chain does vanish. A TREE is not a chain: two leaves
      in different branches have incomparable extents. Which shape WordNet
      actually has decides whether the whole measurement is relevant.

  F3  IS THE FIGURE 4 JOIN DEFICIT JUST TASK DIFFICULTY? The review points
      out the Random baseline — which contains no method at all — also has
      join < meet in 5/5 domains. If the join task is simply harder, the
      deficit says nothing about geometry. Normalize and see what survives.

  F4  EQ. 6 ORIENTATION. The review's own new finding: the paper's soft
      operators (meet = coordinatewise min, join = coordinatewise max) look
      inverted relative to its Proposition 3, and unlike the Definition 7
      dispute this one is in the executed scoring path.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fca  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"

# Figure 4, decoded from the published SVG bar geometry by two agents
# working independently; identical to 3 d.p. NOT printed in the paper.
FIG4 = {
    "domain":  ["WN-Animal", "WN-Plant", "WN-Food", "WN-Event", "WN-Cognition"],
    "meet_random": [0.091, 0.085, 0.094, 0.090, 0.089],
    "join_random": [0.089, 0.083, 0.092, 0.088, 0.087],
    "meet_mean":   [0.321, 0.308, 0.334, 0.298, 0.273],
    "join_mean":   [0.298, 0.287, 0.312, 0.298, 0.299],
    "meet_ours":   [0.547, 0.558, 0.565, 0.505, 0.492],
    "join_ours":   [0.511, 0.525, 0.531, 0.458, 0.470],
}


# ---------------------------------------------------------------- F1 -------

def in_minkowski_sum(v, DA, DB, tol=1e-7):
    """Is v = a + b with a in R(Y_A), b in R(Y_B)?

    R(Y) = {x : D_Y x >= 0}. Feasibility LP in the stacked variable (a, b)
    with the equality a + b = v. Written straight from the definition; it
    shares no code with the region construction it is testing.
    """
    d = len(v)
    A_eq = np.hstack([np.eye(d), np.eye(d)])
    A_ub = np.zeros((DA.shape[0] + DB.shape[0], 2 * d))
    A_ub[: DA.shape[0], :d] = -DA          # -D_A a <= 0  <=>  D_A a >= 0
    A_ub[DA.shape[0]:, d:] = -DB
    r = linprog(np.zeros(2 * d), A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]),
                A_eq=A_eq, b_eq=v, bounds=[(None, None)] * (2 * d),
                method="highs")
    return bool(r.success)


def sample_region(D, rng, n=1, iters=4000):
    """Sample points of R(Y) = {x : Dx >= 0} by rejection, then fall back to
    a nonneg-combination of the dual generators if rejection is starving."""
    d = D.shape[1]
    out = []
    for _ in range(iters):
        x = rng.normal(size=d)
        if (D @ x >= 0).all():
            out.append(x)
            if len(out) >= n:
                return out
    # Fallback: least-squares solve for a strictly interior point.
    for _ in range(iters):
        w = rng.random(D.shape[0]) + 0.05
        x, *_ = np.linalg.lstsq(D, w, rcond=None)
        if (D @ x >= -1e-9).all():
            out.append(x)
            if len(out) >= n:
                break
    return out


def f1_cone_identity(rng) -> dict:
    """Does cone-hull(R(Y_A) u R(Y_B)) equal R(Y_A & Y_B)?"""
    cases = [
        ("independent d=8, |YA u YB|=5, shared=2", 8, 3, 3, 2, False),
        ("independent d=20, |YA u YB|=10, shared=3", 20, 7, 6, 3, False),
        ("independent d=64, |YA u YB|=14, shared=4", 64, 9, 9, 4, False),
        ("DEPENDENT d=2, 4 dirs (k > d)", 2, 2, 2, 0, True),
        ("DEPENDENT d=3, d3 = d1 + d2", 3, 2, 2, 0, True),
    ]
    rows = []
    for label, d, na, nb, shared, force_dep in cases:
        n_tested = n_inside = 0
        for _ in range(12):
            n_tot = na + nb - shared
            if force_dep:
                base = rng.normal(size=(min(d, 2), d))
                D = np.vstack([base, base.sum(0)[None, :],
                               -base[0][None, :]])[:n_tot]
                if D.shape[0] < n_tot:
                    D = np.vstack([D, rng.normal(size=(n_tot - D.shape[0], d))])
            else:
                D = rng.normal(size=(n_tot, d))

            idx_shared = list(range(shared))
            idx_a = idx_shared + list(range(shared, na))
            idx_b = idx_shared + list(range(na, n_tot))
            DA, DB = D[idx_a], D[idx_b]
            D_shared = D[idx_shared] if shared else np.zeros((0, d))

            # Sample from R(Y_A & Y_B) and ask whether each point is in the
            # Minkowski sum R(Y_A) + R(Y_B) = cone-hull of the union.
            pts = sample_region(D_shared if shared else np.zeros((1, d)), rng, n=8)
            for v in pts:
                n_tested += 1
                if in_minkowski_sum(v, DA, DB):
                    n_inside += 1

        rows.append({
            "case": label,
            "linearly_independent": not force_dep,
            "points_tested": n_tested,
            "points_in_conic_hull": n_inside,
            "identity_holds": n_tested > 0 and n_inside == n_tested,
        })
        print(f"  {label:44s} {n_inside}/{n_tested} "
              f"{'EQUAL' if n_inside == n_tested else 'STRICT'}", flush=True)
    return {"cases": rows}


# ---------------------------------------------------------------- F2 -------

def chain_context(n=40) -> fca.Context:
    """Totally nested attributes — the review's benign control."""
    I = np.zeros((n, 12), bool)
    for m in range(12):
        I[: n - m * 3, m] = True
    return fca.Context(I)


def tree_context(depth=4, branch=3) -> fca.Context:
    leaves = []

    def rec(path, dd):
        if dd == depth:
            leaves.append(list(path)); return
        for b in range(branch):
            rec(path + [(dd, b)], dd + 1)

    rec([], 0)
    nodes = sorted({n for L in leaves for n in L})
    return fca.Context(np.array([[n in L for n in nodes] for L in leaves]))


def closure_gap(ctx, rng, cap=400, maxp=20000) -> dict:
    cs = ctx.concepts(max_concepts=cap)
    pairs = list(itertools.combinations(range(len(cs)), 2))
    if len(pairs) > maxp:
        pairs = [pairs[i] for i in rng.choice(len(pairs), maxp, replace=False)]
    notc, ov = 0, []
    for i, j in pairs:
        U = cs[i][0] | cs[j][0]
        C = ctx.close_objs(U)
        if not np.array_equal(U, C):
            notc += 1
        n = int(C.sum())
        ov.append(float((C & ~U).sum()) / n if n else 0.0)
    return {"n_concepts": len(cs), "union_not_closed_frac": notc / len(pairs),
            "overshoot_mean": float(np.mean(ov))}


def f2_shape(rng) -> dict:
    out = {}
    for label, ctx in [("chain_totally_nested", chain_context()),
                       ("tree_depth4_branch3", tree_context(4, 3)),
                       ("tree_depth3_branch4", tree_context(3, 4)),
                       ("iid_bernoulli_p0.3", fca.Context(rng.random((60, 12)) < 0.3)),
                       ("iid_bernoulli_p0.5", fca.Context(rng.random((60, 12)) < 0.5))]:
        out[label] = closure_gap(ctx, rng)
        print(f"  {label:26s} notClosed={out[label]['union_not_closed_frac']:.3f} "
              f"overshoot={out[label]['overshoot_mean']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- F3 -------

def f3_difficulty(_) -> dict:
    """Strip task difficulty out of the Figure 4 join deficit."""
    f = FIG4
    rows = []
    for i, dom in enumerate(f["domain"]):
        r_ratio = f["join_random"][i] / f["meet_random"][i]
        m_ratio = f["join_mean"][i] / f["meet_mean"][i]
        o_ratio = f["join_ours"][i] / f["meet_ours"][i]
        rows.append({
            "domain": dom,
            "raw_gap_ours": round(f["meet_ours"][i] - f["join_ours"][i], 4),
            "join_over_meet_random": round(r_ratio, 4),
            "join_over_meet_mean": round(m_ratio, 4),
            "join_over_meet_ours": round(o_ratio, 4),
            # If the join task were merely harder, every method would be
            # scaled by the same factor as Random. Excess below that is
            # method-specific.
            "excess_deficit_ours_vs_random": round(o_ratio - r_ratio, 4),
            "excess_deficit_mean_vs_random": round(m_ratio - r_ratio, 4),
        })
        print(f"  {dom:14s} random={r_ratio:.3f} mean={m_ratio:.3f} "
              f"ours={o_ratio:.3f}  excess(ours)={o_ratio - r_ratio:+.3f}", flush=True)

    ours_excess = [r["excess_deficit_ours_vs_random"] for r in rows]
    mean_excess = [r["excess_deficit_mean_vs_random"] for r in rows]
    return {
        "per_domain": rows,
        "ours_excess_mean": float(np.mean(ours_excess)),
        "ours_excess_all_negative": all(x < 0 for x in ours_excess),
        "mean_excess_mean": float(np.mean(mean_excess)),
        "mean_excess_all_negative": all(x < 0 for x in mean_excess),
        "note": ("Random contains no method, so its join/meet ratio isolates "
                 "task difficulty. Excess below it is method-specific."),
    }


# ---------------------------------------------------------------- F4 -------

def f4_eq6_orientation(rng, n_trials=400) -> dict:
    """Are the paper's soft meet=min / join=max operators inverted?

    Under the paper's own model the meet's intent is Y_A u Y_B (MORE
    attributes) and the join's intent is Y_A & Y_B (FEWER). Build true meet
    and join extents from a real lattice, take their attribute profiles, and
    see whether coordinatewise min or max reproduces each.
    """
    wins = {"meet_min": 0, "meet_max": 0, "join_min": 0, "join_max": 0}
    cos_acc = {"meet_min": [], "meet_max": [], "join_min": [], "join_max": []}

    for _ in range(n_trials):
        I = rng.random((60, 10)) < 0.4
        ctx = fca.Context(I)
        cs = ctx.concepts(max_concepts=120)
        if len(cs) < 4:
            continue
        i, j = rng.choice(len(cs), 2, replace=False)
        A1, B1 = cs[i]
        A2, B2 = cs[j]

        def profile(A):
            # Mean attribute indicator over the extent — the crisp analogue
            # of Eq. 4's projection profile.
            return I[A].mean(axis=0) if A.any() else np.zeros(I.shape[1])

        pA, pB = profile(A1), profile(A2)
        true_meet = profile(A1 & A2)
        true_join = profile(ctx.close_objs(A1 | A2))

        cmin, cmax = np.minimum(pA, pB), np.maximum(pA, pB)

        def cos(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            return float(u @ v / (nu * nv)) if nu and nv else 0.0

        c = {"meet_min": cos(cmin, true_meet), "meet_max": cos(cmax, true_meet),
             "join_min": cos(cmin, true_join), "join_max": cos(cmax, true_join)}
        for k, v in c.items():
            cos_acc[k].append(v)
        wins["meet_min" if c["meet_min"] > c["meet_max"] else "meet_max"] += 1
        wins["join_min" if c["join_min"] > c["join_max"] else "join_max"] += 1

    n = max(1, wins["meet_min"] + wins["meet_max"])
    res = {
        "n_trials": n,
        "mean_cos": {k: float(np.mean(v)) for k, v in cos_acc.items() if v},
        "meet_max_beats_min_frac": wins["meet_max"] / n,
        "join_min_beats_max_frac": wins["join_min"] / n,
        "paper_uses": "meet=min, join=max",
    }
    res["paper_orientation_looks_inverted"] = (
        res["meet_max_beats_min_frac"] > 0.5 and res["join_min_beats_max_frac"] > 0.5)
    print(f"  mean cos: {res['mean_cos']}")
    print(f"  meet: max beats min in {res['meet_max_beats_min_frac']:.3f} of trials")
    print(f"  join: min beats max in {res['join_min_beats_max_frac']:.3f} of trials")
    return res


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260731)

    print("F1  cone identity: cone-hull(R(Y_A) u R(Y_B)) =?= R(Y_A & Y_B)")
    f1 = f1_cone_identity(rng)
    print("\nF2  context shape: is a WordNet-shaped tree benign like a chain?")
    f2 = f2_shape(rng)
    print("\nF3  Figure 4, difficulty-normalized")
    f3 = f3_difficulty(rng)
    print("\nF4  Eq. 6 orientation")
    f4 = f4_eq6_orientation(rng)

    out = {"F1_cone_identity": f1, "F2_context_shape": f2,
           "F3_figure4_difficulty_normalized": f3, "F4_eq6_orientation": f4}
    (RES / "arm_f_adversarial_response.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RES/'arm_f_adversarial_response.json'}")


if __name__ == "__main__":
    main()
