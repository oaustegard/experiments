"""Arm D — is the paper's geometric join an element of its own lattice?

arXiv:2603.01227 Definition 7 gives the geometric join as a plain set union
of half-space regions:

    A v B := R(Y_A) u R(Y_B)

while its own Appendix B, Proposition 3 gives the order-theoretic join with
the closure operator FCA requires:

    \\/_i (X_i, Y_i) = ( (U_i X_i)'', /\\_i Y_i )

These are not the same object. A set S is a concept extent exactly when it
is closed, S = S''. So the question "is the union-join a lattice element?"
is decidable by a single closure test per pair, with no embeddings, no
probes, and no fitted parameters — which makes it the most robust
measurement in this experiment.

Three candidate join operators are compared:

    J_union    A1 | A2                 Definition 7 / Eq. 6 (the paper's)
    J_closure  (A1 | A2)''             Appendix B Prop. 3 (correct FCA)
    J_top      the intersection of the half-spaces of B1 & B2

J_closure and J_top are provably the same set; computing both independently
is a redundancy check on the implementation, in the spirit of METHODS.md's
"verify with a deliberately disjoint code path".

Runs over every context source available: the LLM-elicited domains, a
WordNet-derived context if arm B produced one, and synthetic controls.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fca  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
CTX_DIR = HERE / "data" / "contexts"
RES_DIR = HERE / "results"

MAX_CONCEPTS = 500
MAX_PAIRS = 20000


def analyse(ctx: fca.Context, label: str, rng) -> dict:
    concepts = ctx.concepts(max_concepts=MAX_CONCEPTS)
    capped = len(concepts) >= MAX_CONCEPTS

    pairs = list(itertools.combinations(range(len(concepts)), 2))
    subsampled = len(pairs) > MAX_PAIRS
    if subsampled:
        sel = rng.choice(len(pairs), MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in sel]

    n_union_closed = 0          # union already a lattice element
    n_meet_closed = 0           # must be all of them (theorem)
    overshoots, phantom_counts, union_sizes = [], [], []
    top_mismatch = 0            # J_closure vs J_top disagreement (must be 0)

    for i, j in pairs:
        A1, B1 = concepts[i]
        A2, B2 = concepts[j]

        # --- meet: bare intersection, should always be closed -------------
        M = A1 & A2
        if np.array_equal(ctx.close_objs(M), M):
            n_meet_closed += 1

        # --- join: three ways ---------------------------------------------
        U = A1 | A2
        J_closure = ctx.close_objs(U)
        J_top = ctx.objs_of(B1 & B2)          # independent route to the same set
        if not np.array_equal(J_closure, J_top):
            top_mismatch += 1

        if np.array_equal(U, J_closure):
            n_union_closed += 1

        n_phantom = int((J_closure & ~U).sum())
        n_closed = int(J_closure.sum())
        phantom_counts.append(n_phantom)
        union_sizes.append(int(U.sum()))
        overshoots.append(n_phantom / n_closed if n_closed else 0.0)

    n = len(pairs)
    ov = np.array(overshoots)
    return {
        "label": label,
        "n_objects": ctx.n_obj,
        "n_attributes": ctx.n_att,
        "density": float(ctx.I.mean()),
        "n_concepts": len(concepts),
        "concepts_capped": capped,
        "n_pairs_examined": n,
        "pairs_subsampled": subsampled,

        # headline: how often is Definition 7's join a lattice element at all
        "union_join_is_lattice_element_frac": n_union_closed / n,
        "union_join_NOT_lattice_element_frac": 1.0 - n_union_closed / n,

        # meet is closed by theorem — this is the known-good calibration side
        "meet_is_lattice_element_frac": n_meet_closed / n,

        # J_closure vs J_top must agree — disjoint-path redundancy check
        "closure_vs_topintent_mismatches": top_mismatch,

        "overshoot_mean": float(ov.mean()),
        "overshoot_median": float(np.median(ov)),
        "overshoot_p90": float(np.percentile(ov, 90)),
        "overshoot_max": float(ov.max()),
        "phantoms_mean": float(np.mean(phantom_counts)),
        "phantoms_max": int(np.max(phantom_counts)),
        "union_size_mean": float(np.mean(union_sizes)),
    }


def synthetic_contexts(rng) -> list[tuple[str, fca.Context]]:
    """Controls: random incidence at several densities, plus a geometric
    realization, so the LLM-elicited numbers have something to sit against."""
    out = []
    for p in (0.2, 0.35, 0.5, 0.65):
        I = rng.random((60, 12)) < p
        out.append((f"synthetic_random_p{p}", fca.Context(I)))

    # A genuinely geometric context: thresholded linear functionals in d dims.
    for d in (32, 8, 4):
        E = rng.normal(size=(60, d))
        V = rng.normal(size=(12, d))
        scores = E @ V.T
        tau = np.quantile(scores, 0.5, axis=0)
        out.append((f"synthetic_halfspace_d{d}", fca.realize(E, V, tau)))
    return out


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []

    for name, ctx in synthetic_contexts(rng):
        rows.append(analyse(ctx, name, rng))
        print(f"[{name}] done", flush=True)

    for p in sorted(CTX_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        ctx = fca.Context(np.array(d["incidence"], dtype=bool))
        rows.append(analyse(ctx, f"llm_{d['domain']}", rng))
        print(f"[llm_{d['domain']}] done", flush=True)

    wn = RES_DIR / "wordnet_context.json"
    if wn.exists():
        d = json.loads(wn.read_text())
        ctx = fca.Context(np.array(d["incidence"], dtype=bool))
        rows.append(analyse(ctx, "wordnet", rng))
        print("[wordnet] done", flush=True)

    # Global sanity: the meet fraction must be exactly 1.0 everywhere and the
    # two join routes must never disagree. Either failing means the code is
    # wrong, not that we discovered something.
    bad = [r["label"] for r in rows if r["meet_is_lattice_element_frac"] != 1.0]
    mism = [r["label"] for r in rows if r["closure_vs_topintent_mismatches"] != 0]
    gates = {
        "meet_always_closed": not bad,
        "meet_gate_failures": bad,
        "closure_routes_agree": not mism,
        "closure_route_mismatches": mism,
    }

    out = {"caps": {"max_concepts": MAX_CONCEPTS, "max_pairs": MAX_PAIRS},
           "gates": gates, "rows": rows}
    (RES_DIR / "arm_d_join_operators.json").write_text(json.dumps(out, indent=2))

    print("\n=== gates ===")
    print(json.dumps(gates, indent=2))
    print("\n=== union-join is NOT a lattice element (fraction of pairs) ===")
    for r in rows:
        print(f"  {r['label']:32s} {r['union_join_NOT_lattice_element_frac']:.3f}"
              f"   mean overshoot {r['overshoot_mean']:.3f}"
              f"   concepts {r['n_concepts']}{' (capped)' if r['concepts_capped'] else ''}")


if __name__ == "__main__":
    main()
