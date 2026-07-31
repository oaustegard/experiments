"""Arm B — WordNet sub-hierarchies + real embeddings: does half-space geometry
recover FCA meets better than FCA joins?

The Lattice Representation Hypothesis (arXiv 2603.01227) claims an LLM
embedding space encodes an FCA concept lattice, with attribute ``m`` true of
object ``g`` iff ``<v_m, e_g> > tau_m``, and lattice operations realized as
half-space intersections.

The asymmetry this arm tests:

    meet extent = A1 & A2            -- a bare intersection.  Under the
                                        half-space model this is exactly an
                                        intersection of half-spaces, so it is
                                        representable *by construction*.
    join extent = (A1 | A2)''        -- the CLOSURE of a union.  A union of
                                        half-space-defined regions is not a
                                        half-space region; the geometry can
                                        only over-approximate it by the
                                        smallest enclosing intersection.

Prediction: join recovery < meet recovery on real embeddings.

Ground truth is WordNet hypernymy (non-LLM, independent of the embedder).
Objects are leaf-ish synsets under 2-4 roots; attributes are ancestor
synsets; ``I[g, m]`` iff ``m`` is a hypernym-ancestor of ``g``.

Protocol notes (METHODS.md):
  * principle 3 -- probes are fit on TRAIN objects only and every lattice
    measurement is made on the held-out TEST sub-context.  The in-sample
    (fit-and-evaluate-on-all) variant is also computed, purely to quantify
    the size of the contamination gap.
  * principle 2 -- two-sided calibration gates: ``assert_meet_closed`` /
    ``assert_join_overshoots`` from fca.py, a known-good chain context that
    must *fail* the overshoot gate, and a synthetic noise sweep that must
    return exactly 1.0 recovery at zero injected noise.
  * multiple seeds, spread reported, every cap named in the JSON.

Run:  python3 src/arm_b_wordnet.py
Out:  results/arm_b_wordnet.json
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fca import (  # noqa: E402
    Context,
    assert_join_overshoots,
    assert_meet_closed,
    coherence,
    join_overshoot,
    realize,
)

RESULTS = HERE.parent / "results" / "arm_b_wordnet.json"

# --------------------------------------------------------------------------
# Caps and knobs -- every one of these is reported in the JSON.  No silent
# truncation anywhere.
# --------------------------------------------------------------------------
CFG = {
    "embedder": "sentence-transformers/all-MiniLM-L6-v2",
    "embed_dim": 384,
    "l2_normalize_embeddings": True,
    "n_seeds": 8,
    "test_frac": 0.40,
    "max_concepts_full": 2000,      # concept enumeration cap, full context
    "max_concepts_test": 1000,      # concept enumeration cap, test sub-context
    "pair_cap": 3000,               # sampled concept pairs per configuration
    "gate_concept_cap": 60,         # concepts fed to the O(n^2) fca.py gates
    "n_attrs_target": 13,
    "n_objects_target": 150,
    "min_train_pos": 3,             # per attribute, per split
    "min_test_pos": 2,
    "split_redraw_limit": 40,
    "noise_sweep_p": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
    "noise_sweep_reps": 5,
    "object_sample_seed": 20260731,
}

TEXT_VARIANTS = ["lemma", "lemma_gloss", "gloss"]
PROBE_METHODS = ["logreg", "lda", "diffmeans", "ctrl_random_dir", "ctrl_shuffled_labels"]
CONTROL_METHODS = {"ctrl_random_dir", "ctrl_shuffled_labels"}


# ==========================================================================
# 1. Ground-truth contexts from WordNet
# ==========================================================================

def _ancestors(syn):
    return set(syn.closure(lambda x: x.hypernyms()))


def _clean_pool(roots, wn):
    """Leaf-ish synsets under ``roots`` with a usable one-or-two-word lemma."""
    pool = set()
    for r in roots:
        pool |= set(wn.synset(r).closure(lambda x: x.hyponyms()))
    keep = []
    for s in pool:
        lem = s.lemmas()[0].name()
        if not lem.replace("_", "").isalpha():
            continue
        if len(lem.split("_")) > 2:
            continue
        if not s.definition():
            continue
        if s.hyponyms():          # leaf-ish only
            continue
        keep.append(s)
    return sorted(keep, key=lambda s: s.name())


def _dedupe_extents(cands, ext):
    """Collapse attributes with byte-identical extents; keep the first by name."""
    seen, reps, dupes = {}, [], {}
    for a in sorted(cands, key=lambda s: s.name()):
        k = ext[a].tobytes()
        if k in seen:
            dupes.setdefault(seen[k].name(), []).append(a.name())
        else:
            seen[k] = a
            reps.append(a)
    return reps, dupes


def _crosses(X, Y):
    """True iff extents X, Y properly overlap (neither nested nor disjoint)."""
    k = int((X & Y).sum())
    return 0 < k < int(X.sum()) and k < int(Y.sum())


def build_context_tree(wn):
    """Context A -- the NAIVE setting: random leaf sample, top-coverage ancestors.

    This is what you get if you do the obvious thing.  It turns out to be a
    bare taxonomy tree (see ``cross_cutting_pairs`` in the output).
    """
    roots = ["animal.n.01", "vehicle.n.01", "plant.n.02"]
    per_root = 50
    rng = random.Random(CFG["object_sample_seed"])
    objs = []
    for r in roots:
        cand = _clean_pool([r], wn)
        objs += rng.sample(cand, per_root)
    objs = sorted(set(objs), key=lambda s: s.name())

    anc = {o: _ancestors(o) for o in objs}
    cnt = Counter()
    for o in objs:
        for a in anc[o]:
            cnt[a] += 1
    n = len(objs)
    cands = [a for a, c in cnt.items() if 0.06 * n <= c <= 0.90 * n]
    ext = {a: np.array([a in anc[o] for o in objs]) for a in cands}
    reps, dupes = _dedupe_extents(cands, ext)
    reps.sort(key=lambda a: -int(ext[a].sum()))
    attrs = reps[: CFG["n_attrs_target"]]
    attrs.sort(key=lambda a: -int(ext[a].sum()))
    return _finish_context("tree", roots, objs, attrs, anc,
                           "random leaf sample per root; attributes = "
                           "distinct-extent ancestors with coverage in "
                           "[0.06, 0.90], top-13 by coverage", dupes)


def build_context_cross(wn):
    """Context B -- attributes chosen to MAXIMIZE cross-cutting (non-nested) pairs.

    Motivation, stated up front because it is a design choice that moves the
    numbers: a pure taxonomy tree makes every non-nested meet empty, which
    makes the meet side of the comparison vacuous.  So here attributes are
    picked greedily to maximize properly-overlapping pairs, and objects are
    stratified over attribute signatures so no signature dominates.
    Still 100% WordNet -- no LLM touches the ground truth.
    """
    roots = ["animal.n.01", "plant.n.02", "food.n.01", "vehicle.n.01"]
    pool = _clean_pool(roots, wn)
    anc = {o: _ancestors(o) for o in pool}
    cnt = Counter()
    for o in pool:
        for a in anc[o]:
            cnt[a] += 1
    n = len(pool)
    cands = [a for a, c in cnt.items() if 0.03 * n <= c <= 0.80 * n]
    ext = {a: np.array([a in anc[o] for o in pool]) for a in cands}
    reps, dupes = _dedupe_extents(cands, ext)

    sel = []
    for _ in range(CFG["n_attrs_target"]):
        best, best_score = None, (-1, -1)
        for a in reps:
            if a in sel:
                continue
            s = sum(1 for b in sel if _crosses(ext[a], ext[b]))
            score = (s, int(ext[a].sum()))
            if score > best_score:
                best_score, best = score, a
        sel.append(best)
    sel.sort(key=lambda a: -int(ext[a].sum()))

    # stratify objects over attribute signatures
    sig = defaultdict(list)
    for o in pool:
        sig[tuple(a in anc[o] for a in sel)].append(o)
    n_sig = len(sig)
    per = max(1, math.ceil(CFG["n_objects_target"] / n_sig))
    rng = random.Random(CFG["object_sample_seed"])
    objs = []
    for k in sorted(sig, key=lambda k: (-len(sig[k]), k)):
        grp = sorted(sig[k], key=lambda s: s.name())
        objs += rng.sample(grp, min(per, len(grp)))
    objs = sorted(set(objs), key=lambda s: s.name())
    return _finish_context("cross", roots, objs, sel, anc,
                           f"greedy cross-cut-maximizing attribute selection "
                           f"over distinct-extent ancestors with coverage in "
                           f"[0.03, 0.80]; objects stratified over the "
                           f"{n_sig} distinct attribute signatures, "
                           f"<= {per} per signature", dupes)


def _finish_context(name, roots, objs, attrs, anc, recipe, dupes):
    I = np.array([[a in anc[o] for a in attrs] for o in objs], dtype=bool)
    n_obj, n_att = I.shape
    cross = sum(1 for i, j in itertools.combinations(range(n_att), 2)
                if _crosses(I[:, i], I[:, j]))
    return {
        "name": name,
        "roots": roots,
        "recipe": recipe,
        "objects": [o.name() for o in objs],
        "object_lemmas": [o.lemmas()[0].name().replace("_", " ") for o in objs],
        "object_glosses": [o.definition() for o in objs],
        "attributes": [a.name() for a in attrs],
        "attribute_prevalence": [float(I[:, i].mean()) for i in range(n_att)],
        "n_objects": int(n_obj),
        "n_attributes": int(n_att),
        "density": float(I.mean()),
        "cross_cutting_attr_pairs": int(cross),
        "total_attr_pairs": int(n_att * (n_att - 1) // 2),
        "collapsed_duplicate_extents": {k: v for k, v in sorted(dupes.items())},
        "I": I,
    }


# ==========================================================================
# 2. Embedding
# ==========================================================================

def make_texts(ctx, variant):
    lem, gl = ctx["object_lemmas"], ctx["object_glosses"]
    if variant == "lemma":
        return list(lem)
    if variant == "lemma_gloss":
        return [f"{l}: {g}" for l, g in zip(lem, gl)]
    if variant == "gloss":
        return list(gl)
    raise ValueError(variant)


def embed_all(contexts, log):
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(CFG["embedder"])
    out = {}
    for cname, ctx in contexts.items():
        out[cname] = {}
        for v in TEXT_VARIANTS:
            texts = make_texts(ctx, v)
            E = model.encode(texts, batch_size=64, show_progress_bar=False,
                             normalize_embeddings=CFG["l2_normalize_embeddings"])
            out[cname][v] = np.asarray(E, dtype=np.float64)
    log(f"embedded in {time.time() - t0:.1f}s")
    return out


# ==========================================================================
# 3. Probes
# ==========================================================================

def fit_probes(E_tr, Y_tr, method, rng):
    """Return (V, tau) with the fca.realize convention: m true iff E@v > tau."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    n_att, d = Y_tr.shape[1], E_tr.shape[1]
    V = np.zeros((n_att, d))
    tau = np.zeros(n_att)

    for m in range(n_att):
        y = Y_tr[:, m]
        if method == "ctrl_shuffled_labels":
            y = y[rng.permutation(len(y))]

        if method == "ctrl_random_dir":
            v = rng.normal(size=d)
            v /= np.linalg.norm(v)
            s = E_tr @ v
            # threshold matched so realized TRAIN prevalence == true TRAIN prevalence
            p = float(Y_tr[:, m].mean())
            t = np.quantile(s, 1.0 - p) if 0.0 < p < 1.0 else (s.max() + 1.0 if p == 0 else s.min() - 1.0)
            V[m], tau[m] = v, float(t)
            continue

        if y.sum() == 0 or y.sum() == len(y):
            V[m] = rng.normal(size=d) * 1e-9
            tau[m] = -1.0 if y.all() else 1.0
            continue

        if method in ("logreg", "ctrl_shuffled_labels"):
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(E_tr, y)
            V[m] = clf.coef_[0]
            tau[m] = -float(clf.intercept_[0])
        elif method == "lda":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(E_tr, y)
            V[m] = clf.coef_[0]
            tau[m] = -float(clf.intercept_[0])
        elif method == "diffmeans":
            v = E_tr[y].mean(0) - E_tr[~y].mean(0)
            nv = np.linalg.norm(v)
            v = v / nv if nv > 0 else v
            s = E_tr @ v
            # threshold picked on TRAIN only, maximizing balanced accuracy
            cuts = np.unique(s)
            mids = (cuts[:-1] + cuts[1:]) / 2 if len(cuts) > 1 else cuts
            best_t, best_b = float(np.median(s)), -1.0
            for t in mids:
                pred = s > t
                tpr = (pred & y).sum() / max(1, y.sum())
                tnr = (~pred & ~y).sum() / max(1, (~y).sum())
                b = 0.5 * (tpr + tnr)
                if b > best_b:
                    best_b, best_t = b, float(t)
            V[m], tau[m] = v, best_t
        else:
            raise ValueError(method)
    return V, tau


def probe_quality(E_te, Y_te, V, tau):
    from sklearn.metrics import roc_auc_score
    S = E_te @ V.T
    P = S > tau[None, :]
    per = []
    for m in range(Y_te.shape[1]):
        y = Y_te[:, m]
        auc = None
        if 0 < y.sum() < len(y):
            auc = float(roc_auc_score(y, S[:, m]))
        per.append({
            "auc": auc,
            "acc": float((P[:, m] == y).mean()),
            "n_pos": int(y.sum()),
            "n": int(len(y)),
        })
    return per


# ==========================================================================
# 4. Set-recovery metrics
# ==========================================================================

def setmetrics(pred, true, n):
    inter = int((pred & true).sum())
    up = int((pred | true).sum())
    p, t = int(pred.sum()), int(true.sum())
    sym = int((pred ^ true).sum())
    return {
        "jaccard": 1.0 if up == 0 else inter / up,
        "precision": 1.0 if p == 0 else inter / p,
        "recall": 1.0 if t == 0 else inter / t,
        "err_frac": sym / n,                       # symmetric difference / |G|
        "rel_err": sym / t if t else (0.0 if p == 0 else float(p)),
        "true_empty": t == 0,
        "true_size": t,
        "pred_size": p,
    }


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def lattice_recovery(ctx_true, ctx_hat, concepts, pair_rng, pair_cap):
    """Meet / union / join recovery over sampled pairs of TRUE concepts.

    The intent B_i is the specification; the extent is the prediction.  For a
    true concept (A_i, B_i), the geometry's extent is ctx_hat.objs_of(B_i).
    """
    n = ctx_true.n_obj
    k = len(concepts)
    all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    n_all = len(all_pairs)
    if n_all > pair_cap:
        idx = pair_rng.choice(n_all, size=pair_cap, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    hat_ext = [ctx_hat.objs_of(B) for _, B in concepts]

    rows = {"extent": [], "meet": [], "union": [], "join": []}
    intrinsic, join_err, meet_err = [], [], []
    n_meet_empty = 0
    identity_violations = {"meet": 0, "join": 0}
    # (n_halfspaces, op, jaccard, err_frac, rel_err, true_size) -- the size control
    intent_rows = []
    k_meet, k_join = [], []

    for i, _ in enumerate(concepts):
        rows["extent"].append(setmetrics(hat_ext[i], concepts[i][0], n))

    for (i, j) in pairs:
        A1, B1 = concepts[i]
        A2, B2 = concepts[j]
        H1, H2 = hat_ext[i], hat_ext[j]

        t_meet = A1 & A2
        t_union = A1 | A2
        t_join = ctx_true.close_objs(t_union)

        # --- representability identity -------------------------------------
        # For genuine concepts,   meet extent = objs_of(B1 | B2)
        #                    and  join extent = objs_of(B1 & B2).
        # BOTH are plain intersections of half-spaces under the LRH model, so
        # neither is intrinsically unrepresentable.  The meet just intersects
        # MORE half-spaces (|B1|B2| >= |B1&B2|).  Verified here rather than
        # assumed.
        Bm, Bj = B1 | B2, B1 & B2
        if not np.array_equal(ctx_true.objs_of(Bm), t_meet):
            identity_violations["meet"] += 1
        if not np.array_equal(ctx_true.objs_of(Bj), t_join):
            identity_violations["join"] += 1
        k_meet.append(int(Bm.sum()))
        k_join.append(int(Bj.sum()))

        h_meet = H1 & H2                      # exact half-space intersection
        h_union = H1 | H2                     # not a lattice element; reference
        h_join = ctx_hat.close_objs(H1 | H2)  # geometry's closure of the union

        mm = setmetrics(h_meet, t_meet, n)
        mu = setmetrics(h_union, t_union, n)
        mj = setmetrics(h_join, t_join, n)
        rows["meet"].append(mm)
        rows["union"].append(mu)
        rows["join"].append(mj)
        intent_rows.append((int(Bm.sum()), "meet", mm))
        intent_rows.append((int(Bj.sum()), "join", mj))

        ov = join_overshoot(ctx_true, A1, A2)
        intrinsic.append(ov["overshoot"])
        join_err.append(mj["err_frac"])
        meet_err.append(mm["err_frac"])
        if t_meet.sum() == 0:
            n_meet_empty += 1

    def summarize(key, only_nonempty=False):
        rs = rows[key]
        if only_nonempty:
            rs = [r for r in rs if not r["true_empty"]]
        if not rs:
            return None
        return {
            "jaccard": _agg([r["jaccard"] for r in rs]),
            "precision": _agg([r["precision"] for r in rs]),
            "recall": _agg([r["recall"] for r in rs]),
            "err_frac": _agg([r["err_frac"] for r in rs]),
            "rel_err": _agg([r["rel_err"] for r in rs]),
            "true_size": _agg([float(r["true_size"]) for r in rs]),
            "n_items": len(rs),
        }

    corr = None
    if len(intrinsic) > 2 and np.std(intrinsic) > 0 and np.std(join_err) > 0:
        corr = float(np.corrcoef(intrinsic, join_err)[0, 1])

    # ---- size control: recovery of objs_of(B) as a function of |B| ---------
    # Meet and join targets are both of the form objs_of(B); if the meet/join
    # gap is entirely a function of |B|, the two operations are the same
    # problem at different constraint counts and the asymmetry thesis has no
    # residual content.
    by_k = defaultdict(lambda: {"meet": [], "join": []})
    for k_, op, m in intent_rows:
        by_k[k_][op].append(m)
    size_control = []
    for k_ in sorted(by_k):
        row = {"n_halfspaces": k_}
        for op in ("meet", "join"):
            ms = by_k[k_][op]
            row[op] = None if not ms else {
                "n": len(ms),
                "jaccard": _agg([m["jaccard"] for m in ms]),
                "err_frac": _agg([m["err_frac"] for m in ms]),
                "true_size": _agg([float(m["true_size"]) for m in ms]),
            }
        size_control.append(row)
    # paired comparison restricted to (k, op) cells where BOTH ops occur
    both = [r for r in size_control if r["meet"] and r["join"]]
    matched = None
    if both:
        dm = [r["meet"]["jaccard"]["mean"] - r["join"]["jaccard"]["mean"] for r in both]
        de = [r["join"]["err_frac"]["mean"] - r["meet"]["err_frac"]["mean"] for r in both]
        matched = {
            "n_matched_k_cells": len(both),
            "k_values": [r["n_halfspaces"] for r in both],
            "meet_minus_join_jaccard_at_matched_k": _agg(dm),
            "join_minus_meet_errfrac_at_matched_k": _agg(de),
        }

    return {
        "n_halfspaces_meet": _agg([float(x) for x in k_meet]),
        "n_halfspaces_join": _agg([float(x) for x in k_join]),
        "identity_violations": identity_violations,
        "size_control_by_halfspace_count": size_control,
        "size_matched_meet_vs_join": matched,
        "n_concepts": k,
        "n_pairs_total": n_all,
        "n_pairs_used": len(pairs),
        "pair_subsampled": n_all > pair_cap,
        "n_meet_empty": n_meet_empty,
        "extent": summarize("extent"),
        "meet_all": summarize("meet"),
        "meet_nonempty": summarize("meet", only_nonempty=True),
        "union_all": summarize("union"),
        "join_all": summarize("join"),
        "join_nonempty": summarize("join", only_nonempty=True),
        "intrinsic_overshoot": _agg(intrinsic),
        "corr_intrinsic_overshoot_vs_join_err": corr,
        "mean_meet_err_frac": float(np.mean(meet_err)),
        "mean_join_err_frac": float(np.mean(join_err)),
    }


# ==========================================================================
# 5. Calibration gates
# ==========================================================================

def run_gates(ctx_true, log):
    """Two-sided gates: fca.py's own pair, plus a known-good context that must
    FAIL the overshoot gate (proving the gate discriminates rather than always
    passing)."""
    out = {}
    cs = ctx_true.concepts(max_concepts=CFG["gate_concept_cap"])
    out["n_concepts_checked"] = len(cs)
    try:
        out["meet_closed_pairs_checked"] = assert_meet_closed(ctx_true, cs)
        out["meet_closed_gate"] = "pass"
    except AssertionError as e:
        out["meet_closed_gate"] = f"FAIL: {e}"
    try:
        out["join_overshoot_pairs"] = assert_join_overshoots(ctx_true, cs)
        out["join_overshoot_gate"] = "pass"
    except AssertionError as e:
        out["join_overshoot_gate"] = f"FAIL(vacuous context): {e}"

    # known-bad side for the GATE itself: a nested chain has every union already
    # closed, so assert_join_overshoots MUST raise.  If it does not, the gate is
    # a rubber stamp.
    chain = Context(np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool))
    cc = chain.concepts()
    try:
        assert_join_overshoots(chain, cc)
        out["gate_discriminates_on_chain"] = "FAIL: gate passed a chain context"
    except AssertionError:
        out["gate_discriminates_on_chain"] = "pass (chain correctly rejected)"
    return out


def noise_sweep(ctx_true, concepts, rng, pair_cap):
    """Synthetic probe error with a KNOWN rate.

    This is the decomposition control: it holds the lattice fixed and varies
    only the incidence noise, so any meet/join divergence here is caused by
    the closure step and not by anything embedding-specific.  At p = 0 both
    recoveries must be exactly 1.0 -- that is the harness's own calibration.
    """
    I = ctx_true.I
    n_obj, n_att = I.shape
    out = []
    for p in CFG["noise_sweep_p"]:
        meets, joins, unions = [], [], []
        me, je, ue, mk = [], [], [], []
        for rep in range(CFG["noise_sweep_reps"] if p > 0 else 1):
            flip = rng.random((n_obj, n_att)) < p
            ctx_hat = Context(I ^ flip)
            r = lattice_recovery(ctx_true, ctx_hat, concepts, rng, pair_cap)
            meets.append(r["meet_all"]["jaccard"]["mean"])
            joins.append(r["join_all"]["jaccard"]["mean"])
            unions.append(r["union_all"]["jaccard"]["mean"])
            me.append(r["meet_all"]["err_frac"]["mean"])
            je.append(r["join_all"]["err_frac"]["mean"])
            ue.append(r["union_all"]["err_frac"]["mean"])
            if r["size_matched_meet_vs_join"]:
                mk.append(r["size_matched_meet_vs_join"]
                          ["meet_minus_join_jaccard_at_matched_k"]["mean"])
        out.append({
            "flip_rate": p,
            "reps": len(meets),
            "meet_jaccard": _agg(meets),
            "union_jaccard": _agg(unions),
            "join_jaccard": _agg(joins),
            "meet_err_frac": _agg(me),
            "union_err_frac": _agg(ue),
            "join_err_frac": _agg(je),
            "meet_minus_join": float(np.mean(meets) - np.mean(joins)),
            "union_minus_join": float(np.mean(unions) - np.mean(joins)),
            "meet_minus_join_at_matched_halfspace_count": _agg(mk),
        })
    return out


# ==========================================================================
# 6. Driver
# ==========================================================================

def draw_split(n_obj, I, seed):
    """Random object split, redrawn until every attribute is estimable."""
    rng = np.random.default_rng(seed)
    for attempt in range(CFG["split_redraw_limit"]):
        perm = rng.permutation(n_obj)
        n_te = int(round(CFG["test_frac"] * n_obj))
        te, tr = perm[:n_te], perm[n_te:]
        ok = True
        for m in range(I.shape[1]):
            ytr, yte = I[tr, m], I[te, m]
            if ytr.sum() < CFG["min_train_pos"] or (~ytr).sum() < CFG["min_train_pos"]:
                ok = False
                break
            if yte.sum() < CFG["min_test_pos"] or (~yte).sum() < CFG["min_test_pos"]:
                ok = False
                break
        if ok:
            return tr, te, attempt
    return tr, te, -1  # last draw, flagged


def main():
    t_start = time.time()
    logs = []

    def log(msg):
        print(msg, flush=True)
        logs.append(msg)

    from nltk.corpus import wordnet as wn
    wn.synsets("dog")  # force corpus load

    log("building WordNet contexts ...")
    contexts = {"tree": build_context_tree(wn), "cross": build_context_cross(wn)}
    for c in contexts.values():
        log(f"  {c['name']}: {c['n_objects']} obj x {c['n_attributes']} att, "
            f"density {c['density']:.3f}, cross-cutting attr pairs "
            f"{c['cross_cutting_attr_pairs']}/{c['total_attr_pairs']}")

    emb = embed_all(contexts, log)

    out = {
        "arm": "B — WordNet sub-hierarchies, real embeddings",
        "paper": "arXiv 2603.01227 (Lattice Representation Hypothesis)",
        "config": CFG,
        "text_variants": TEXT_VARIANTS,
        "probe_methods": PROBE_METHODS,
        "control_methods": sorted(CONTROL_METHODS),
        "contexts": {},
    }

    for cname, ctx in contexts.items():
        I = ctx["I"]
        n_obj, n_att = I.shape
        ctx_full = Context(I)

        log(f"[{cname}] gates + full-context structure ...")
        gates = run_gates(ctx_full, log)
        cs_full = ctx_full.concepts(max_concepts=CFG["max_concepts_full"])
        full_capped = len(cs_full) >= CFG["max_concepts_full"]

        # intrinsic (zero-probe-error) overshoot of the TRUE lattice
        prng = np.random.default_rng(11)
        k = len(cs_full)
        allp = [(i, j) for i in range(k) for j in range(i + 1, k)]
        if len(allp) > CFG["pair_cap"]:
            sel = prng.choice(len(allp), size=CFG["pair_cap"], replace=False)
            sp = [allp[i] for i in sel]
        else:
            sp = allp
        ov = [join_overshoot(ctx_full, cs_full[i][0], cs_full[j][0]) for i, j in sp]
        intrinsic_full = {
            "overshoot": _agg([o["overshoot"] for o in ov]),
            "phantoms": _agg([float(o["phantoms"]) for o in ov]),
            "frac_pairs_with_phantoms": float(np.mean([o["phantoms"] > 0 for o in ov])),
            "n_pairs_used": len(sp),
            "n_pairs_total": len(allp),
            "pair_subsampled": len(allp) > CFG["pair_cap"],
        }

        log(f"[{cname}] noise sweep (synthetic known-rate probe error) ...")
        nrng = np.random.default_rng(101)
        # noise sweep on a size-matched sub-context so it is comparable to the
        # held-out measurements
        n_te = int(round(CFG["test_frac"] * n_obj))
        sub = np.sort(np.random.default_rng(5).permutation(n_obj)[:n_te])
        ctx_sub = Context(I[sub])
        cs_sub = ctx_sub.concepts(max_concepts=CFG["max_concepts_test"])
        sweep = noise_sweep(ctx_sub, cs_sub, nrng, CFG["pair_cap"])
        for s in sweep:
            log(f"   p={s['flip_rate']:.2f}  meet J={s['meet_jaccard']['mean']:.4f} "
                f"union J={s['union_jaccard']['mean']:.4f} "
                f"join J={s['join_jaccard']['mean']:.4f}")

        # ---- held-out probe + lattice measurement ------------------------
        per_config = defaultdict(list)     # (variant, method) -> list of seed results
        per_attr = defaultdict(lambda: defaultdict(list))  # (v,meth) -> attr -> aucs
        concept_cache = {}

        for si in range(CFG["n_seeds"]):
            seed = 1000 + si
            tr, te, attempt = draw_split(n_obj, I, seed)
            if attempt < 0:
                log(f"  WARNING seed {seed}: split redraw limit hit, "
                    f"some attribute may be degenerate")
            ctx_true_te = Context(I[te])
            if si not in concept_cache:
                cs_te = ctx_true_te.concepts(max_concepts=CFG["max_concepts_test"])
                concept_cache[si] = (cs_te, len(cs_te) >= CFG["max_concepts_test"])
            cs_te, te_capped = concept_cache[si]

            for v in TEXT_VARIANTS:
                E = emb[cname][v]
                E_tr, E_te = E[tr], E[te]
                Y_tr, Y_te = I[tr], I[te]
                for meth in PROBE_METHODS:
                    rng = np.random.default_rng(seed * 31 + hash(meth) % 1000)
                    V, tau = fit_probes(E_tr, Y_tr, meth, rng)
                    pq = probe_quality(E_te, Y_te, V, tau)
                    for ai, a in enumerate(ctx["attributes"]):
                        per_attr[(v, meth)][a].append(pq[ai])
                    ctx_hat_te = realize(E_te, V, tau)
                    cell_acc = float((ctx_hat_te.I == Y_te).mean())
                    tp = int((ctx_hat_te.I & Y_te).sum())
                    fp = int((ctx_hat_te.I & ~Y_te).sum())
                    fn = int((~ctx_hat_te.I & Y_te).sum())
                    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                    prng2 = np.random.default_rng(seed * 7 + 3)
                    rec = lattice_recovery(ctx_true_te, ctx_hat_te, cs_te,
                                           prng2, CFG["pair_cap"])
                    rec["cell_accuracy"] = cell_acc
                    rec["cell_f1"] = f1
                    rec["coherence"] = coherence(V)
                    rec["mean_auc"] = _agg([p["auc"] for p in pq])
                    rec["mean_probe_acc"] = _agg([p["acc"] for p in pq])
                    rec["test_concepts_capped"] = te_capped
                    per_config[(v, meth)].append(rec)

        # ---- aggregate across seeds --------------------------------------
        agg = {}
        for (v, meth), runs in per_config.items():
            def pull(path):
                vals = []
                for r in runs:
                    cur = r
                    for p in path:
                        if cur is None:
                            break
                        cur = cur.get(p) if isinstance(cur, dict) else None
                    vals.append(cur if not isinstance(cur, dict) else cur.get("mean"))
                return _agg(vals)

            entry = {
                "n_seeds": len(runs),
                "mean_heldout_auc": pull(["mean_auc"]),
                "mean_heldout_probe_acc": pull(["mean_probe_acc"]),
                "cell_accuracy": pull(["cell_accuracy"]),
                "cell_f1": pull(["cell_f1"]),
                "direction_coherence": pull(["coherence"]),
                "n_concepts_test": pull(["n_concepts"]),
                "n_pairs_used": pull(["n_pairs_used"]),
                "pair_subsampled_any": any(r["pair_subsampled"] for r in runs),
                "test_concepts_capped_any": any(r["test_concepts_capped"] for r in runs),
                "frac_pairs_with_empty_true_meet": _agg(
                    [r["n_meet_empty"] / max(1, r["n_pairs_used"]) for r in runs]),
                "extent_jaccard": pull(["extent", "jaccard"]),
                "meet_jaccard_all": pull(["meet_all", "jaccard"]),
                "meet_jaccard_nonempty": pull(["meet_nonempty", "jaccard"]),
                "meet_precision": pull(["meet_all", "precision"]),
                "meet_recall": pull(["meet_all", "recall"]),
                "meet_err_frac": pull(["meet_all", "err_frac"]),
                "meet_rel_err": pull(["meet_all", "rel_err"]),
                "meet_true_size": pull(["meet_all", "true_size"]),
                "union_jaccard_all": pull(["union_all", "jaccard"]),
                "union_err_frac": pull(["union_all", "err_frac"]),
                "join_jaccard_all": pull(["join_all", "jaccard"]),
                "join_jaccard_nonempty": pull(["join_nonempty", "jaccard"]),
                "join_precision": pull(["join_all", "precision"]),
                "join_recall": pull(["join_all", "recall"]),
                "join_err_frac": pull(["join_all", "err_frac"]),
                "join_rel_err": pull(["join_all", "rel_err"]),
                "join_true_size": pull(["join_all", "true_size"]),
                "n_halfspaces_meet": pull(["n_halfspaces_meet"]),
                "n_halfspaces_join": pull(["n_halfspaces_join"]),
                "identity_violations_meet": int(sum(
                    r["identity_violations"]["meet"] for r in runs)),
                "identity_violations_join": int(sum(
                    r["identity_violations"]["join"] for r in runs)),
                "meet_minus_join_jaccard_at_matched_k": pull(
                    ["size_matched_meet_vs_join",
                     "meet_minus_join_jaccard_at_matched_k"]),
                "join_minus_meet_errfrac_at_matched_k": pull(
                    ["size_matched_meet_vs_join",
                     "join_minus_meet_errfrac_at_matched_k"]),
                "intrinsic_overshoot": pull(["intrinsic_overshoot"]),
                "corr_intrinsic_overshoot_vs_join_err": pull(
                    ["corr_intrinsic_overshoot_vs_join_err"]),
            }
            # per-seed paired deltas -- the headline comparison
            d_mj = [r["meet_all"]["jaccard"]["mean"] - r["join_all"]["jaccard"]["mean"]
                    for r in runs]
            d_uj = [r["union_all"]["jaccard"]["mean"] - r["join_all"]["jaccard"]["mean"]
                    for r in runs]
            d_ej = [r["extent"]["jaccard"]["mean"] - r["join_all"]["jaccard"]["mean"]
                    for r in runs]
            d_err = [r["join_all"]["err_frac"]["mean"] - r["meet_all"]["err_frac"]["mean"]
                     for r in runs]
            entry["delta_meet_minus_join"] = _agg(d_mj)
            entry["delta_union_minus_join"] = _agg(d_uj)
            entry["delta_extent_minus_join"] = _agg(d_ej)
            entry["delta_join_minus_meet_err_frac"] = _agg(d_err)
            entry["meet_beats_join_in_n_of_n_seeds"] = [
                int(sum(1 for x in d_mj if x > 0)), len(d_mj)]
            entry["join_err_exceeds_meet_err_in_n_of_n_seeds"] = [
                int(sum(1 for x in d_err if x > 0)), len(d_err)]
            # keep the full half-space-count table only for the primary probe,
            # to bound JSON size
            if meth == "logreg":
                entry["size_control_by_halfspace_count_seed0"] = \
                    runs[0]["size_control_by_halfspace_count"]
            agg[f"{v}|{meth}"] = entry

        # per-attribute held-out quality, logreg only (kept small)
        pa = {}
        for v in TEXT_VARIANTS:
            pa[v] = {}
            for a, rows in per_attr[(v, "logreg")].items():
                pa[v][a] = {
                    "auc": _agg([r["auc"] for r in rows]),
                    "acc": _agg([r["acc"] for r in rows]),
                    "prevalence_test": _agg([r["n_pos"] / r["n"] for r in rows]),
                }

        # in-sample (contaminated) contrast, logreg / lemma_gloss only
        insample = {}
        for v in TEXT_VARIANTS:
            E = emb[cname][v]
            rng = np.random.default_rng(0)
            V, tau = fit_probes(E, I, "logreg", rng)
            pq = probe_quality(E, I, V, tau)
            ctx_hat = realize(E, V, tau)
            cs = ctx_full.concepts(max_concepts=CFG["max_concepts_test"])
            rec = lattice_recovery(ctx_full, ctx_hat, cs,
                                   np.random.default_rng(2), CFG["pair_cap"])
            insample[v] = {
                "in_sample_auc": _agg([p["auc"] for p in pq]),
                "in_sample_cell_accuracy": float((ctx_hat.I == I).mean()),
                "meet_jaccard_all": rec["meet_all"]["jaccard"]["mean"],
                "union_jaccard_all": rec["union_all"]["jaccard"]["mean"],
                "join_jaccard_all": rec["join_all"]["jaccard"]["mean"],
                "extent_jaccard": rec["extent"]["jaccard"]["mean"],
                "n_concepts": rec["n_concepts"],
                "n_pairs_used": rec["n_pairs_used"],
            }

        out["contexts"][cname] = {
            "spec": {k: v for k, v in ctx.items() if k != "I"},
            "calibration_gates": gates,
            "full_context_concepts": len(cs_full),
            "full_context_concepts_capped": full_capped,
            "intrinsic_join_overshoot_true_lattice": intrinsic_full,
            "noise_sweep": sweep,
            "heldout": agg,
            "per_attribute_heldout_logreg": pa,
            "in_sample_contaminated_logreg": insample,
        }
        log(f"[{cname}] done ({time.time() - t_start:.0f}s elapsed)")

    out["runtime_seconds"] = round(time.time() - t_start, 1)
    out["log"] = logs

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: (
            o.item() if isinstance(o, np.generic) else
            o.tolist() if isinstance(o, np.ndarray) else str(o)))
    tmp.replace(RESULTS)
    print(f"\nwrote {RESULTS}  ({RESULTS.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
