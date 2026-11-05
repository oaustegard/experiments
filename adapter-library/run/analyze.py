"""Combine the CPU and GPU arms into results/summary.json and check P1-P7.

    python3 -I run/analyze.py          # after unzipping out/al_out.zip into this directory
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import GPU_SEEDS, ROOT, SIZES, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "out"
acc = lambda name: json.loads((RES / f"{name}.json").read_text())["test"]["accuracy"]
probs = lambda name: json.loads((OUT / "probs" / f"{name}.json").read_text())
KNN = json.loads((OUT / "knn_probs.json").read_text())
TE = {w: examples(load_cases(w, "test")) for w in WORKFLOWS}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def score(w, p):
    return score_examples(TE[w], p)[0]["accuracy"]


def mix(a, b, lam):
    return [[lam * x + (1 - lam) * y for x, y in zip(pa, pb)] for pa, pb in zip(a, b)]


def main():
    s = {}
    # S: spikes
    s["S"] = {w: {k: [acc(f"S-{k}-{w}-s{i}") for i in (0, 1, 2)] for k in ("lora", "full")} for w in WORKFLOWS}
    lora = np.mean([np.mean(v["lora"]) for v in s["S"].values()])
    full = np.mean([np.mean(v["full"]) for v in s["S"].values()])
    s["S_mean"] = {"lora": lora, "full": full}

    # G fallback at n = 0, including the routed spike per test case
    routed = json.loads((OUT / "routed_to.json").read_text())
    fb = {}
    for w in WORKFLOWS:
        lib = [x for x in WORKFLOWS if x != w]
        P = {x: probs(f"G-fallback-{w}-spike-{x}") for x in lib}
        r = {x: acc(f"G-fallback-{w}-spike-{x}") for x in lib}
        r["merge"] = acc(f"G-fallback-{w}-merge")
        per = [P[routed[w][e["case"]]][i] for i, e in enumerate(TE[w])]
        r["routed"] = score(w, per)
        fb[w] = r
    s["fallback"] = fb

    # G growth: scratch / warm / full / knn / mixes, mean over seeds
    g = {}
    for w in WORKFLOWS:
        for n in SIZES[1:]:
            row = {k: [] for k in ("scratch", "warm", "full", "knn", "mix_scratch", "mix_warm")}
            sweep = {f"{k}@{l}": [] for k in ("scratch", "warm") for l in LAMBDAS}
            for seed in GPU_SEEDS:
                kp = KNN[f"{w}|s{seed}|n{n}"]
                row["knn"].append(score(w, kp))
                for kind in ("scratch", "warm", "full"):
                    row[kind].append(acc(f"G-{kind}-{w}-n{n}-s{seed}"))
                for kind in ("scratch", "warm"):
                    p = probs(f"G-{kind}-{w}-n{n}-s{seed}")
                    row[f"mix_{kind}"].append(score(w, mix(p, kp, 0.5)))
                    for l in LAMBDAS:
                        sweep[f"{kind}@{l}"].append(score(w, mix(p, kp, l)))
            g[f"{w}|n{n}"] = {**{k: float(np.mean(v)) for k, v in row.items()},
                              "post_hoc_lambda_sweep": {k: float(np.mean(v)) for k, v in sweep.items()}}
    s["G"] = g
    M = lambda n, k: float(np.mean([g[f"{w}|n{n}"][k] for w in WORKFLOWS]))
    s["G_mean"] = {f"n{n}": {k: M(n, k) for k in ("scratch", "warm", "full", "knn", "mix_scratch", "mix_warm")}
                   for n in SIZES[1:]}

    # predictions
    gm = s["G_mean"]
    fb_best = max(max(v.values()) for v in fb.values())
    s["predictions"] = {
        "P1_lora_within_0.02_of_full": abs(lora - full) <= 0.02,
        "P3_fallback_max_le_0.50": fb_best <= 0.50,
        "P4_knn_beats_scratch_n16": gm["n16"]["knn"] > gm["n16"]["scratch"],
        "P4_scratch_beats_knn_n300_by_0.03": gm["n300"]["scratch"] - gm["n300"]["knn"] >= 0.03,
        "P5_warm_beats_scratch_n16_by_0.03": gm["n16"]["warm"] - gm["n16"]["scratch"] >= 0.03,
        "P5_warm_scratch_within_0.02_n300": abs(gm["n300"]["warm"] - gm["n300"]["scratch"]) <= 0.02,
        "P6_mix_beats_both_by_0.01_n_le_64": all(
            gm[f"n{n}"][f"mix_{k}"] - max(gm[f"n{n}"][k], gm[f"n{n}"]["knn"]) >= 0.01
            for n in (16, 64) for k in ("scratch", "warm")),
        "P7_scratch_n300_within_0.03_of_S_lora": abs(gm["n300"]["scratch"] - lora) <= 0.03,
    }
    routing = json.loads((RES / "routing.json").read_text())
    s["predictions"]["P2_routing_ge_0.98_gap_auroc_ge_0.95"] = routing["accuracy"] >= 0.98 and all(
        v["auroc"] >= 0.95 for v in routing["gap"].values())
    (RES / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({"S_mean": s["S_mean"], "fallback": fb, "G_mean": gm, "predictions": s["predictions"]}, indent=1))


if __name__ == "__main__":
    main()
