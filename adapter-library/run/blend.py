"""Round 2: learn the spike/kNN blend weight per decision, leave-one-workflow-out for the gate.

    python3 -I run/blend.py

p = lam * p_spike + (1 - lam) * p_knn, lam = sigmoid(w . f + b). Features use only the revealed cases
and the two models' outputs (see PLAN.md, round 2). For held-out W the gate is fit on the other three
workflows' growth runs and applied to W unchanged. Reads round 1's saved distributions
(out/probs/G-*.json, out/knn_probs.json, out/emb.npz); writes results/blend.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import GPU_SEEDS, ROOT, WORKFLOWS, examples, load_cases, revealed, score_examples  # noqa: E402

K, TAU = 8, 0.05  # same kNN as round 1 (run/knn.py)
NS, KINDS = [16, 64, 300], ["scratch", "warm"]
FEATS = ["top1_cos", "knn_loo_acc", "knn_conf", "spike_conf", "tv_disagree", "log_n", "warm"]
L2 = 1e-3
OUT, RES = ROOT / "out", ROOT / "results"


def knn_vote(keys, golds, q):
    s = keys @ q
    top = np.argsort(-s)[: min(K, len(s))]
    w = np.exp((s[top] - s[top].max()) / TAU)
    return (w[:, None] * golds[top]).sum(0) / w.sum()


def loo_acc(keys, golds):
    """kNN accuracy on the revealed cases themselves, each predicted from the others."""
    ok = []
    for j in range(len(keys)):
        m = np.arange(len(keys)) != j
        p = knn_vote(keys[m], golds[m], keys[j])
        ok.append(p.argmax() == golds[j].argmax())
    return float(np.mean(ok))


def build_rows():
    z = np.load(OUT / "emb.npz")
    E = dict(zip(z["ids"], z["X"]))
    KNN = json.loads((OUT / "knn_probs.json").read_text())
    TE = {w: examples(load_cases(w, "test")) for w in WORKFLOWS}
    rows = []
    for w in WORKFLOWS:
        for seed in GPU_SEEDS:
            for n in NS:
                by_q = {}
                for e in examples(revealed(w, seed, n)):
                    by_q.setdefault(e["qname"], []).append((E[e["case"]], np.array(e["gold"])))
                stats = {}
                for q, v in by_q.items():
                    keys, golds = np.stack([k for k, _ in v]), np.stack([g for _, g in v])
                    stats[q] = (keys, loo_acc(keys, golds))
                pk = KNN[f"{w}|s{seed}|n{n}"]
                for kind in KINDS:
                    ps = json.loads((OUT / "probs" / f"G-{kind}-{w}-n{n}-s{seed}.json").read_text())
                    for i, e in enumerate(TE[w]):
                        keys, la = stats[e["qname"]]
                        a, b = np.array(ps[i]), np.array(pk[i])
                        f = [float((keys @ E[e["case"]]).max()), la, b.max(), a.max(),
                             0.5 * np.abs(a - b).sum(), math.log(n), float(kind == "warm")]
                        rows.append({"w": w, "seed": seed, "n": n, "kind": kind, "i": i, "f": f,
                                     "ps": a, "pk": b, "g": np.array(e["gold"])})
    return rows, TE


def pad(rows, key, L):
    out = np.zeros((len(rows), L))
    for r, x in enumerate(rows):
        out[r, : len(x[key])] = x[key]
    return torch.tensor(out)


def fit(rows, use_feats, mu, sd):
    L = max(len(r["g"]) for r in rows)
    G, PS, PK = pad(rows, "g", L), pad(rows, "ps", L), pad(rows, "pk", L)
    F = torch.tensor((np.array([r["f"] for r in rows]) - mu) / sd)
    w = torch.zeros(F.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    params = [w, b] if use_feats else [b]
    opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lam = torch.sigmoid((F @ w if use_feats else 0) + b)[:, None]
        p = lam * PS + (1 - lam) * PK
        loss = -(G * torch.log(p + 1e-9)).sum(1).mean() + (L2 * (w ** 2).sum() if use_feats else 0)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy() if use_feats else np.zeros(F.shape[1]), float(b.detach())


def lam_of(rows, wb, mu, sd):
    w, b = wb
    F = (np.array([r["f"] for r in rows]) - mu) / sd
    return 1 / (1 + np.exp(-(F @ w + b)))


def acc(TE, w, rows, lam, metric="accuracy"):
    probs = [(l * r["ps"] + (1 - l) * r["pk"]).tolist() for r, l in zip(rows, lam)]
    return score_examples(TE[w], probs)[0][metric]


def main():
    rows, TE = build_rows()
    print(f"{len(rows)} rows", flush=True)
    out, coefs = {}, {}
    for w in WORKFLOWS:
        tr = [r for r in rows if r["w"] != w]
        Ftr = np.array([r["f"] for r in tr])
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        gate, const = fit(tr, True, mu, sd), fit(tr, False, mu, sd)
        coefs[w] = {"weights": dict(zip(FEATS, [round(float(x), 3) for x in gate[0]])), "bias": round(gate[1], 3),
                    "const_lambda": round(1 / (1 + math.exp(-const[1])), 3)}
        for n in NS:
            for kind in KINDS:
                for seed in GPU_SEEDS:
                    sel = [r for r in rows if (r["w"], r["n"], r["kind"], r["seed"]) == (w, n, kind, seed)]
                    sel.sort(key=lambda r: r["i"])
                    lg = lam_of(sel, gate, mu, sd)
                    res = {"spike": acc(TE, w, sel, np.ones(len(sel))), "knn": acc(TE, w, sel, np.zeros(len(sel))),
                           "fixed": acc(TE, w, sel, np.full(len(sel), 0.5)),
                           "const": acc(TE, w, sel, lam_of(sel, const, mu, sd)),
                           "gate": acc(TE, w, sel, lg), "gate_mean_lambda": float(lg.mean())}
                    for nm, lam in (("spike", np.ones(len(sel))), ("knn", np.zeros(len(sel))),
                                    ("fixed", np.full(len(sel), 0.5)), ("const", lam_of(sel, const, mu, sd)), ("gate", lg)):
                        res[f"{nm}_kl"] = acc(TE, w, sel, lam, "kl_from_gold")
                    grid = {round(l, 2): acc(TE, w, sel, np.full(len(sel), l)) for l in np.linspace(0, 1, 21)}
                    res["oracle_const"] = max(grid.values())
                    out[f"{w}|n{n}|{kind}|s{seed}"] = res
        print(w, coefs[w], flush=True)
    keys = ["spike", "knn", "fixed", "const", "gate", "oracle_const", "gate_mean_lambda",
            "spike_kl", "knn_kl", "fixed_kl", "const_kl", "gate_kl"]
    mean = {f"n{n}": {kind: {k: float(np.mean([out[f"{w}|n{n}|{kind}|s{s}"][k] for w in WORKFLOWS for s in GPU_SEEDS]))
                             for k in keys} for kind in KINDS} for n in NS}
    by_w = {w: {f"n{n}": {kind: {k: float(np.mean([out[f"{w}|n{n}|{kind}|s{s}"][k] for s in GPU_SEEDS]))
                                 for k in keys} for kind in KINDS} for n in NS} for w in WORKFLOWS}
    m = mean
    pred = {
        "P8a_gate_beats_better_component_n16_n64": all(
            m[f"n{n}"][k]["gate"] > max(m[f"n{n}"][k]["spike"], m[f"n{n}"][k]["knn"]) for n in (16, 64) for k in KINDS),
        "P8b_gate_within_0.005_of_spike_n300": all(m["n300"][k]["gate"] >= m["n300"][k]["spike"] - 0.005 for k in KINDS),
        "P9_gate_beats_fixed_every_n": all(m[f"n{n}"][k]["gate"] > m[f"n{n}"][k]["fixed"] for n in NS for k in KINDS),
        "P10_gate_beats_const_by_0.01_n16_or_n64": any(
            m[f"n{n}"][k]["gate"] - m[f"n{n}"][k]["const"] >= 0.01 for n in (16, 64) for k in KINDS),
        "P11_scratch_gain_largest_at_n16": max(NS, key=lambda n: m[f"n{n}"]["scratch"]["gate"] - m[f"n{n}"]["scratch"]["spike"]) == 16,
    }
    res = {"features": FEATS, "l2": L2, "mean": mean, "by_workflow": by_w, "gate_coefs": coefs,
           "predictions": {k: bool(v) for k, v in pred.items()}, "runs": out}
    (RES / "blend.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"mean": mean, "predictions": res["predictions"]}, indent=1))


if __name__ == "__main__":
    main()
