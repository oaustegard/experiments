"""Round 3: choose the spike/kNN blend weight from out-of-fold predictions on W's own revealed cases.

    python3 -I run/blend_cv.py      # after the CV-* arms of run/lora.py (out/probs/CV-*.json)

Arms (PLAN.md round 3): cv_lambda (per question, the lambda that minimizes out-of-fold soft cross-
entropy), cv_select (spike or kNN, whichever had higher out-of-fold accuracy), gate2 (round 2's gate
plus out-of-fold spike and kNN accuracy as features, leave-one-workflow-out). Writes results/cv_blend.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import FEATS, acc, build_rows, fit, knn_vote, lam_of  # noqa: E402
from shared import GPU_SEEDS, ROOT, WORKFLOWS, examples, revealed  # noqa: E402

NS, KINDS, FOLDS, EPS = [16, 64], ["scratch", "warm"], 4, 0.01
GRID = [round(x, 1) for x in np.linspace(0, 1, 11)]
OUT, RES = ROOT / "out", ROOT / "results"


def ce(g, p):
    return -float(np.sum(g * np.log(p + 1e-9)))


def per_question_stats(E):
    """(W, n, seed, kind) -> {qname: (spike_oof_acc, knn_oof_acc, cv_lambda)}."""
    stats = {}
    for w in WORKFLOWS:
        for seed in GPU_SEEDS:
            for n in NS:
                rev = revealed(w, seed, n)
                fold = {c["id"]: i % FOLDS for i, c in enumerate(rev)}
                ex = examples(rev)
                pk = []
                for e in ex:
                    others = [o for o in ex if o["qname"] == e["qname"] and fold[o["case"]] != fold[e["case"]]]
                    keys = np.stack([E[o["case"]] for o in others])
                    golds = np.stack([np.array(o["gold"]) for o in others])
                    p = knn_vote(keys, golds, E[e["case"]])
                    pk.append((1 - EPS) * p + EPS / len(p))
                for kind in KINDS:
                    ps = [np.array(p) for p in json.loads((OUT / "probs" / f"CV-{kind}-{w}-n{n}-s{seed}.json").read_text())]
                    q_stats = {}
                    for q in sorted({e["qname"] for e in ex}):
                        idx = [i for i, e in enumerate(ex) if e["qname"] == q]
                        g = [np.array(ex[i]["gold"]) for i in idx]
                        sa = np.mean([ps[i].argmax() == gg.argmax() for i, gg in zip(idx, g)])
                        ka = np.mean([pk[i].argmax() == gg.argmax() for i, gg in zip(idx, g)])
                        lam = min(GRID, key=lambda l: sum(ce(gg, l * ps[i] + (1 - l) * pk[i]) for i, gg in zip(idx, g)))
                        q_stats[q] = (float(sa), float(ka), lam)
                    stats[(w, n, seed, kind)] = q_stats
    return stats


def main():
    rows, TE = build_rows()
    rows = [r for r in rows if r["n"] in NS]
    z = np.load(OUT / "emb.npz")
    E = dict(zip(z["ids"], z["X"]))
    stats = per_question_stats(E)
    for r in rows:
        sa, ka, _ = stats[(r["w"], r["n"], r["seed"], r["kind"])][TE[r["w"]][r["i"]]["qname"]]
        r["f"] = r["f"] + [sa, ka]
    r2 = json.loads((RES / "blend.json").read_text())
    out = {}
    for w in WORKFLOWS:
        tr = [r for r in rows if r["w"] != w]
        F = np.array([r["f"] for r in tr])
        mu, sd = F.mean(0), F.std(0) + 1e-9
        gate2 = fit(tr, True, mu, sd)
        for n in NS:
            for kind in KINDS:
                for seed in GPU_SEEDS:
                    sel = sorted([r for r in rows if (r["w"], r["n"], r["kind"], r["seed"]) == (w, n, kind, seed)],
                                 key=lambda r: r["i"])
                    qs = stats[(w, n, seed, kind)]
                    qn = [TE[w][r["i"]]["qname"] for r in sel]
                    lam_cv = np.array([qs[q][2] for q in qn])
                    lam_sel = np.array([1.0 if qs[q][0] >= qs[q][1] else 0.0 for q in qn])
                    out[f"{w}|n{n}|{kind}|s{seed}"] = {
                        "spike": acc(TE, w, sel, np.ones(len(sel))), "knn": acc(TE, w, sel, np.zeros(len(sel))),
                        "cv_lambda": acc(TE, w, sel, lam_cv), "cv_select": acc(TE, w, sel, lam_sel),
                        "gate2": acc(TE, w, sel, lam_of(sel, gate2, mu, sd)),
                        "round2_gate": r2["runs"][f"{w}|n{n}|{kind}|s{seed}"]["gate"],
                        "cv_lambda_kl": acc(TE, w, sel, lam_cv, "kl_from_gold"),
                        "spike_kl": acc(TE, w, sel, np.ones(len(sel)), "kl_from_gold"),
                        "mean_cv_lambda": float(lam_cv.mean()),
                        "spike_oof_acc": float(np.mean([v[0] for v in qs.values()])),
                        "knn_oof_acc": float(np.mean([v[1] for v in qs.values()])),
                    }
    keys = list(next(iter(out.values())).keys())
    M = lambda flt: {k: float(np.mean([v[k] for kk, v in out.items() if flt(kk)])) for k in keys}
    mean = {f"n{n}": {kind: M(lambda kk, n=n, kind=kind: f"|n{n}|{kind}|" in kk) for kind in KINDS} for n in NS}
    by_w = {w: {f"n{n}": {kind: M(lambda kk, w=w, n=n, kind=kind: kk.startswith(f"{w}|n{n}|{kind}|")) for kind in KINDS}
                for n in NS} for w in WORKFLOWS}

    # P16 diagnostic: out-of-fold (kNN - spike) accuracy difference vs the test difference, per question, n = 16
    xs, ys = [], []
    for (w, n, seed, kind), qs in stats.items():
        if n != 16:
            continue
        sel = sorted([r for r in rows if (r["w"], r["n"], r["kind"], r["seed"]) == (w, n, kind, seed)], key=lambda r: r["i"])
        for q, (sa, ka, _) in qs.items():
            rq = [r for r in sel if TE[w][r["i"]]["qname"] == q]
            ts = np.mean([r["ps"].argmax() == r["g"].argmax() for r in rq])
            tk = np.mean([r["pk"].argmax() == r["g"].argmax() for r in rq])
            xs.append(ka - sa)
            ys.append(tk - ts)
    r16 = float(np.corrcoef(xs, ys)[0, 1])

    m = mean
    pred = {
        "P12_cv_lambda_beats_spike_n16_n64_both": all(m[f"n{n}"][k]["cv_lambda"] > m[f"n{n}"][k]["spike"] for n in NS for k in KINDS),
        "P13_cv_lambda_beats_round2_gate_by_0.01_n16_any": any(
            m["n16"][k]["cv_lambda"] - m["n16"][k]["round2_gate"] >= 0.01 for k in KINDS),
        "P14_agent_traces_n16_warm_within_0.01_of_spike": by_w["agent_trace_observability"]["n16"]["warm"]["cv_lambda"]
        >= by_w["agent_trace_observability"]["n16"]["warm"]["spike"] - 0.01,
        "P15_gate2_beats_round2_gate_by_0.01_n16_any": any(m["n16"][k]["gate2"] - m["n16"][k]["round2_gate"] >= 0.01 for k in KINDS),
        "P16_oof_diff_corr_ge_0.4_n16": r16 >= 0.4,
    }
    res = {"features_gate2": FEATS + ["spike_oof_acc", "knn_oof_acc"], "grid": GRID, "folds": FOLDS,
           "p16_corr_n16": r16, "mean": mean, "by_workflow": by_w,
           "predictions": {k: bool(v) for k, v in pred.items()}, "runs": out}
    (RES / "cv_blend.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"mean": mean, "p16_corr_n16": r16, "predictions": res["predictions"]}, indent=1))


if __name__ == "__main__":
    main()
