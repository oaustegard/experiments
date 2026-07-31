"""Arm E — does the meet/join gap survive on realistic context shapes?

Two objections to Arm A deserve a direct answer rather than a rebuttal in
prose:

  (a) VACUITY. "The union of two extents is rarely closed" might be a
      generic fact about lattices, saying nothing about embeddings. If a
      WordNet-shaped hypernym tree — the paper's own setting — turned out
      to be benign, the whole result would be an artifact of random
      contexts.

  (b) UNREALISTIC GEOMETRY. Arm A draws embeddings iid standard normal and
      attribute directions independently. Real LLM embedding spaces are
      anisotropic (a dominant mean direction) and their attribute
      directions are strongly correlated. Arm A's own coherence sweep shows
      overshoot falling sharply as coherence rises, so real spaces might
      sit in the benign regime.

This arm builds both adversarial cases explicitly and measures them.
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
RES = HERE / "results"

MAX_CONCEPTS = 400
MAX_PAIRS = 20000
SEEDS = (0, 1, 2, 3, 4)


def measure(ctx: fca.Context, rng) -> dict:
    cs = ctx.concepts(max_concepts=MAX_CONCEPTS)
    pairs = list(itertools.combinations(range(len(cs)), 2))
    subsampled = len(pairs) > MAX_PAIRS
    if subsampled:
        pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS, replace=False)]

    not_lattice, ov, meet_bad = 0, [], 0
    for i, j in pairs:
        A1, A2 = cs[i][0], cs[j][0]
        M = A1 & A2
        if not np.array_equal(ctx.close_objs(M), M):
            meet_bad += 1
        U = A1 | A2
        C = ctx.close_objs(U)
        if not np.array_equal(U, C):
            not_lattice += 1
        n = int(C.sum())
        ov.append(float((C & ~U).sum()) / n if n else 0.0)

    return {
        "n_concepts": len(cs),
        "concepts_capped": len(cs) >= MAX_CONCEPTS,
        "pairs_subsampled": subsampled,
        "n_pairs": len(pairs),
        "union_join_not_lattice_frac": not_lattice / len(pairs),
        "meet_not_closed_count": meet_bad,          # must stay 0
        "overshoot_mean": float(np.mean(ov)),
        "overshoot_median": float(np.median(ov)),
    }


def tree_context(depth: int = 4, branch: int = 3) -> fca.Context:
    """A pure hypernym tree: objects are leaves, attributes are ancestors.

    This is the shape of a WordNet sub-hierarchy — precisely the data the
    paper builds its formal contexts from.
    """
    leaves: list[list] = []

    def rec(path, d):
        if d == depth:
            leaves.append(list(path))
            return
        for b in range(branch):
            rec(path + [(d, b)], d + 1)

    rec([], 0)
    nodes = sorted({n for L in leaves for n in L})
    I = np.array([[n in L for n in nodes] for L in leaves])
    return fca.Context(I)


def anisotropic_realization(rng, n_obj=60, d=64, n_att=12, r=0.0, aniso=3.0):
    """Embeddings with a dominant mean direction and correlated attribute
    directions — a crude stand-in for real LLM embedding geometry.

    ``r`` mixes each attribute direction toward a single shared direction,
    which is the knob that drives mutual coherence up.
    """
    E = rng.normal(size=(n_obj, d)) + aniso * rng.normal(size=d)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    shared = rng.normal(size=d)
    V = np.sqrt(1 - r) * rng.normal(size=(n_att, d)) + np.sqrt(r) * shared
    tau = np.quantile(E @ V.T, 0.5, axis=0)
    return fca.realize(E, V, tau), fca.coherence(V)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(label, builder, extra=None):
        per = [measure(builder(np.random.default_rng(s)),
                       np.random.default_rng(1000 + s)) for s in SEEDS]
        keys = [k for k, v in per[0].items() if isinstance(v, float)]
        rec = {"label": label, "n_seeds": len(SEEDS)}
        rec.update({k: {"mean": float(np.mean([p[k] for p in per])),
                        "std": float(np.std([p[k] for p in per]))} for k in keys})
        rec["n_concepts"] = per[0]["n_concepts"]
        rec["concepts_capped"] = per[0]["concepts_capped"]
        rec["meet_not_closed_total"] = sum(p["meet_not_closed_count"] for p in per)
        if extra:
            rec.update(extra)
        rows.append(rec)
        print(f"  {label:44s} notLattice={rec['union_join_not_lattice_frac']['mean']:.3f}"
              f"  overshoot={rec['overshoot_mean']['mean']:.3f}", flush=True)

    # (a) the vacuity objection: is a WordNet-shaped tree benign?
    print("context shape:")
    add("tree_hypernym_depth4_branch3", lambda rng: tree_context(4, 3))
    add("tree_hypernym_depth3_branch4", lambda rng: tree_context(3, 4))

    def tree_plus(rng, k=4):
        t = tree_context(4, 3)
        return fca.Context(np.hstack([t.I, rng.random((t.n_obj, k)) < 0.5]))

    add("tree_plus_4_crosscutting_attrs", tree_plus)
    add("random_p0.5_control", lambda rng: fca.Context(rng.random((60, 12)) < 0.5))

    # (b) the geometry objection: does realistic anisotropy rescue the join?
    print("embedding geometry:")
    for r in (0.0, 0.3, 0.5, 0.8, 0.9):
        coh_holder = {}

        def build(rng, r=r, holder=coh_holder):
            ctx, coh = anisotropic_realization(rng, r=r)
            holder["coherence"] = coh
            return ctx

        add(f"anisotropic_dir_corr_r{r}", build, extra=None)
        rows[-1]["coherence"] = coh_holder.get("coherence")

    gates = {"meet_never_broken": all(r["meet_not_closed_total"] == 0 for r in rows)}
    out = {"caps": {"max_concepts": MAX_CONCEPTS, "max_pairs": MAX_PAIRS},
           "seeds": list(SEEDS), "gates": gates, "rows": rows}
    (RES / "arm_e_context_shape.json").write_text(json.dumps(out, indent=2))
    print("\ngates:", gates)
    print(f"wrote {RES/'arm_e_context_shape.json'}")


if __name__ == "__main__":
    main()
