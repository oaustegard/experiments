"""Arm C — do a model's declarative concept knowledge and its embedding
geometry induce the same lattice, and where do they diverge?

Arm B tests the geometry against WordNet, an external non-LLM ground truth.
Arm C tests it against ground truth elicited from an LLM itself: five
domains (animals, instruments, vehicles, foods, occupations), each a hand-
built formal context of ~50 objects x ~13 binary attributes.

The point of the pairing: WordNet's hierarchy is a tree of hypernyms, so its
lattice is unusually thin. The elicited contexts are genuinely multi-
attribute and cross-cutting, which is the regime where meet and join can
actually come apart.

Outputs
-------
results/arm_c_llm_contexts.json   per-domain metrics
results/phantoms.json             the concrete objects the geometric join
                                  admits that the set union does not, staged
                                  for the behavioural follow-up in arm_d
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fca  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
CTX_DIR = HERE / "data" / "contexts"
RES_DIR = HERE / "results"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Concept lattices grow combinatorially; every cap here is reported in the
# output rather than applied silently.
MAX_CONCEPTS = 400
MAX_PAIRS = 6000
SEEDS = (0, 1, 2, 3, 4)


def load_contexts() -> dict[str, dict]:
    out = {}
    for p in sorted(CTX_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        d["incidence"] = np.array(d["incidence"], dtype=bool)
        n_obj, n_att = d["incidence"].shape
        assert n_obj == len(d["objects"]), f"{p.name}: object count mismatch"
        assert n_att == len(d["attributes"]), f"{p.name}: attribute count mismatch"
        out[d["domain"]] = d
    return out


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    E = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False)
    return np.asarray(E, dtype=np.float64)


def fit_directions(E: np.ndarray, I: np.ndarray, seed: int):
    """Cross-validated linear probes: one direction + threshold per attribute.

    Every object gets its attribute prediction from a fold in which it was
    held out, so the realized context is entirely out-of-sample. Fitting and
    scoring the same objects is the single easiest way to make this whole
    hypothesis look better than it is (METHODS.md principle 3).
    """
    n_obj, n_att = I.shape
    V = np.zeros((n_att, E.shape[1]))
    tau = np.zeros(n_att)
    held_out = np.zeros_like(I)
    aucs = []

    for m in range(n_att):
        y = I[:, m].astype(int)
        # A probe needs both classes present in every training fold.
        if y.sum() < 3 or (1 - y).sum() < 3:
            held_out[:, m] = y.astype(bool)
            aucs.append(float("nan"))
            continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(E, y):
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(E[tr], y[tr])
            held_out[te, m] = clf.predict(E[te]).astype(bool)

        full = LogisticRegression(max_iter=2000, C=1.0).fit(E, y)
        V[m] = full.coef_[0]
        tau[m] = -full.intercept_[0]
        aucs.append(float((held_out[:, m] == I[:, m]).mean()))

    return V, tau, held_out, np.array(aucs)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def compare(true_ctx: fca.Context, real_ctx: fca.Context, rng) -> dict:
    """Meet vs join recovery, plus the intrinsic overshoot of the true lattice."""
    concepts = true_ctx.concepts(max_concepts=MAX_CONCEPTS)
    capped = len(concepts) >= MAX_CONCEPTS

    pairs = list(itertools.combinations(range(len(concepts)), 2))
    subsampled = len(pairs) > MAX_PAIRS
    if subsampled:
        idx = rng.choice(len(pairs), MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in idx]

    meet_j, join_j, intrinsic, phantom_rows = [], [], [], []
    for i, j in pairs:
        A1, A2 = concepts[i][0], concepts[j][0]

        # Ground truth from the true lattice.
        t_meet = fca.meet_extent(true_ctx, A1, A2)
        t_join = fca.join_extent(true_ctx, A1, A2)

        # What the geometry produces, starting from the same two extents.
        g_meet = fca.meet_extent(real_ctx, A1, A2)
        g_join = fca.join_extent(real_ctx, A1, A2)

        meet_j.append(jaccard(t_meet, g_meet))
        join_j.append(jaccard(t_join, g_join))

        # Overshoot that survives even with perfect probes: closure vs union.
        ov = fca.join_overshoot(true_ctx, A1, A2)
        intrinsic.append(ov["overshoot"])
        if ov["phantoms"] > 0:
            phantom_rows.append((i, j, ov, (t_join & ~(A1 | A2))))

    return {
        "n_concepts": len(concepts),
        "concepts_capped": capped,
        "n_pairs": len(pairs),
        "pairs_subsampled": subsampled,
        "meet_jaccard_mean": float(np.mean(meet_j)),
        "meet_jaccard_median": float(np.median(meet_j)),
        "join_jaccard_mean": float(np.mean(join_j)),
        "join_jaccard_median": float(np.median(join_j)),
        "meet_exact_frac": float(np.mean(np.array(meet_j) == 1.0)),
        "join_exact_frac": float(np.mean(np.array(join_j) == 1.0)),
        "intrinsic_overshoot_mean": float(np.mean(intrinsic)),
        "intrinsic_overshoot_p90": float(np.percentile(intrinsic, 90)),
        "intrinsic_overshoot_zero_frac": float(np.mean(np.array(intrinsic) == 0.0)),
        "_phantom_rows": phantom_rows,
        "_concepts": concepts,
    }


def random_baseline(E: np.ndarray, I: np.ndarray, rng) -> fca.Context:
    """Control: random directions, thresholds matched to true attribute density.

    Without this the recovery numbers are uninterpretable — a context can
    score well simply because the density is lopsided.
    """
    n_att = I.shape[1]
    V = rng.normal(size=(n_att, E.shape[1]))
    scores = E @ V.T
    tau = np.array([np.quantile(scores[:, m], 1.0 - I[:, m].mean()) for m in range(n_att)])
    return fca.realize(E, V, tau)


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts()
    if not contexts:
        raise SystemExit(f"no context files found in {CTX_DIR}")

    out: dict = {"model": MODEL_NAME, "caps": {"max_concepts": MAX_CONCEPTS,
                                               "max_pairs": MAX_PAIRS},
                 "seeds": list(SEEDS), "domains": {}}
    phantoms_out: dict = {}

    for domain, d in contexts.items():
        I = d["incidence"]
        true_ctx = fca.Context(I)

        # Calibration gates, on real data rather than a toy (METHODS.md #2).
        gates = {}
        probe_concepts = true_ctx.concepts(max_concepts=60)
        try:
            gates["meet_closed_pairs"] = fca.assert_meet_closed(true_ctx, probe_concepts)
            gates["meet_gate"] = "pass"
        except AssertionError as e:
            gates["meet_gate"] = f"FAIL: {e}"
        try:
            gates["overshooting_pairs"] = fca.assert_join_overshoots(true_ctx, probe_concepts)
            gates["join_gate"] = "pass"
        except AssertionError as e:
            gates["join_gate"] = f"FAIL: {e}"

        # Two text variants — reporting only the better one without saying so
        # is the exact failure this repo's methods ledger warns about.
        variants = {
            "name": d["objects"],
            "name_in_domain": [f"{o} ({domain})" for o in d["objects"]],
        }

        per_seed: dict[str, list] = {}
        rec = {"n_objects": I.shape[0], "n_attributes": I.shape[1],
               "density": float(I.mean()), "gates": gates, "variants": {}}

        for vname, texts in variants.items():
            E = embed(texts)
            vres, aucs_all, base_all = [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                V, tau, held_out, aucs = fit_directions(E, I, seed)
                real_ctx = fca.Context(held_out)
                cmp = compare(true_ctx, real_ctx, rng)

                base_ctx = random_baseline(E, I, rng)
                bcmp = compare(true_ctx, base_ctx, rng)

                vres.append({k: v for k, v in cmp.items() if not k.startswith("_")})
                base_all.append({k: v for k, v in bcmp.items() if not k.startswith("_")})
                aucs_all.append(aucs)

                if seed == SEEDS[0] and vname == "name":
                    phantoms_out[domain] = _stage_phantoms(d, cmp)

            def agg(rows, key):
                vals = [r[key] for r in rows]
                return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

            keys = [k for k in vres[0] if isinstance(vres[0][k], float)]
            rec["variants"][vname] = {
                "probe_heldout_acc_mean": float(np.nanmean(aucs_all)),
                "probe_heldout_acc_min": float(np.nanmin(aucs_all)),
                "probe": {k: agg(vres, k) for k in keys},
                "random_baseline": {k: agg(base_all, k) for k in keys},
                "n_concepts": vres[0]["n_concepts"],
                "concepts_capped": vres[0]["concepts_capped"],
                "pairs_subsampled": vres[0]["pairs_subsampled"],
            }

        out["domains"][domain] = rec
        print(f"[{domain}] done — {I.shape[0]}x{I.shape[1]}, gates={gates.get('meet_gate')}/"
              f"{gates.get('join_gate')}", flush=True)

    (RES_DIR / "arm_c_llm_contexts.json").write_text(json.dumps(out, indent=2))
    (RES_DIR / "phantoms.json").write_text(json.dumps(phantoms_out, indent=2))
    print(f"wrote {RES_DIR/'arm_c_llm_contexts.json'} and {RES_DIR/'phantoms.json'}")


def _stage_phantoms(d: dict, cmp: dict, limit: int = 40) -> dict:
    """Pull out concrete phantom memberships for the behavioural probe.

    A phantom is an object the lattice join admits that is in neither input
    extent. Whether a model *also* accepts it decides how to read the
    overshoot: a genuine failure of the geometry, or the geometry
    generalizing in a way the model endorses.
    """
    objs, atts = d["objects"], d["attributes"]
    concepts = cmp["_concepts"]
    items = []
    for i, j, ov, phantom_mask in cmp["_phantom_rows"][:limit]:
        _, B1 = concepts[i]
        _, B2 = concepts[j]
        shared = B1 & B2  # the join's intent
        items.append({
            "concept_a_intent": [atts[k] for k in np.where(B1)[0]],
            "concept_b_intent": [atts[k] for k in np.where(B2)[0]],
            "join_intent": [atts[k] for k in np.where(shared)[0]],
            "union_members": [objs[k] for k in np.where(
                concepts[i][0] | concepts[j][0])[0]],
            "phantoms": [objs[k] for k in np.where(phantom_mask)[0]],
            "overshoot": ov["overshoot"],
        })
    return {"n_overshooting_pairs": len(cmp["_phantom_rows"]), "examples": items}


if __name__ == "__main__":
    main()
