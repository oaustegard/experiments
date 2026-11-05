"""Round 4b analysis: router spike vs kNN router on 12 confusable families (PLAN.md round 4b).

    python3 -I run/router.py

Reads out/probs/R-router-F*.json, out/probs/R-sub-*.json, out/probs/S-lora-<w>-s0.json and out/emb.npz;
writes results/router.json with per-fold and mean routing accuracy, gap AUROC, zero-shot routing to the
held-out family, kNN with 4 examples of it, and downstream decision accuracy under each routing.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn import auroc  # noqa: E402
from shared import FAM_IDS, FAM_WORKFLOW, ROOT, WORKFLOWS, examples, family_map, load_cases, router_fold, score_examples  # noqa: E402

K, SHOTS = 8, 4
OUT, RES = ROOT / "out", ROOT / "results"


def knn_route(E, keys, labs, cid):
    top = np.argsort(-(keys @ E[cid]))[:K]
    return Counter(labs[top]).most_common(1)[0][0]


def main():
    fm = family_map()
    z = np.load(OUT / "emb.npz")
    E = dict(zip(z["ids"], z["X"]))
    train = [c for w in WORKFLOWS for c in load_cases(w, "train")]
    TE = {w: examples(load_cases(w, "test")) for w in WORKFLOWS}
    P = lambda name: json.loads((OUT / "probs" / f"{name}.json").read_text())
    sub = {f: P(f"R-sub-{f}") for f in FAM_IDS}
    whole = {w: P(f"S-lora-{w}-s0") for w in WORKFLOWS}
    idx = {w: {} for w in WORKFLOWS}
    for w in WORKFLOWS:
        for i, e in enumerate(TE[w]):
            idx[w].setdefault(e["case"], []).append(i)

    def downstream(route):
        """route: case id -> family or 'none'; decision accuracy over the routed cases."""
        probs, exs = {w: [] for w in WORKFLOWS}, {w: [] for w in WORKFLOWS}
        for cid, r in route.items():
            w = FAM_WORKFLOW[fm[cid]]
            src = sub[r] if r in FAM_WORKFLOW and FAM_WORKFLOW[r] == w else whole[w]
            for i in idx[w][cid]:
                probs[w].append(src[i])
                exs[w].append(TE[w][i])
        allp = [p for w in WORKFLOWS for p in probs[w]]
        alle = [e for w in WORKFLOWS for e in exs[w]]
        return score_examples(alle, allp)[0]["accuracy"]

    folds = []
    for i in range(len(FAM_IDS)):
        r = P(f"R-router-F{i:02d}")
        F, G = router_fold(i)
        lib = r["library"]
        cases, lab = r["cases"], [fm[c] for c in r["cases"]]
        A = {c: r["options_A"][int(np.argmax(p))] for c, p in zip(cases, r["probs_A"])}
        B = {c: r["options_B"][int(np.argmax(p))] for c, p in zip(cases, r["probs_B"])}
        pnone = {c: p[r["options_A"].index("none")] for c, p in zip(cases, r["probs_A"])}
        libc = [c for c, l in zip(cases, lab) if l in lib]
        fc = [c for c, l in zip(cases, lab) if l == F]
        gc = [c for c, l in zip(cases, lab) if l == G]
        store = [c for c in train if fm[c["id"]] in lib]
        keys, labs = np.stack([E[c["id"]] for c in store]), np.array([fm[c["id"]] for c in store])
        kroute = {c: knn_route(E, keys, labs, c) for c in libc}
        fshots = sorted([c["id"] for c in train if fm[c["id"]] == F])
        random.Random(f"shots-{i}").shuffle(fshots)
        keys4 = np.concatenate([keys, np.stack([E[c] for c in fshots[:SHOTS]])])
        labs4 = np.concatenate([labs, np.array([F] * SHOTS)])
        gap_knn = {c: 1 - float((keys @ E[c]).max()) for c in libc + fc}
        by_w = {}
        for w in WORKFLOWS:
            cw = [c for c in libc if FAM_WORKFLOW[fm[c]] == w]
            if cw:
                by_w[w] = {"router": float(np.mean([A[c] == fm[c] for c in cw])),
                           "knn": float(np.mean([kroute[c] == fm[c] for c in cw])), "n": len(cw)}
        folds.append({
            "F": F, "G": G, "F_workflow": FAM_WORKFLOW[F],
            "routing_router": float(np.mean([A[c] == fm[c] for c in libc])),
            "routing_knn": float(np.mean([kroute[c] == fm[c] for c in libc])),
            "routing_by_workflow": by_w,
            "gap_auroc_router": auroc(np.array([pnone[c] for c in fc]), np.array([pnone[c] for c in libc])),
            "gap_auroc_knn": auroc(np.array([gap_knn[c] for c in fc]), np.array([gap_knn[c] for c in libc])),
            "none_rate_on_G": float(np.mean([A[c] == "none" for c in gc])),
            "none_rate_on_F": float(np.mean([A[c] == "none" for c in fc])),
            "zero_shot_to_F": float(np.mean([B[c] == F for c in fc])),
            "routing_router_B": float(np.mean([B[c] == fm[c] for c in libc])),
            "knn_4shot_to_F": float(np.mean([knn_route(E, keys4, labs4, c) == F for c in fc])),
            "down_router": downstream({c: A[c] for c in libc}),
            "down_knn": downstream(kroute),
            "down_oracle": downstream({c: fm[c] for c in libc}),
            "down_whole": downstream({c: "none" for c in libc}),
            "n_lib": len(libc), "n_F": len(fc),
        })
        f = folds[-1]
        print(f"F{i:02d} {F:18s} route {f['routing_router']:.3f}/{f['routing_knn']:.3f} gap {f['gap_auroc_router']:.2f}/"
              f"{f['gap_auroc_knn']:.2f} zs {f['zero_shot_to_F']:.2f} knn4 {f['knn_4shot_to_F']:.2f} "
              f"down r/k/o/w {f['down_router']:.3f}/{f['down_knn']:.3f}/{f['down_oracle']:.3f}/{f['down_whole']:.3f}", flush=True)
    keys = [k for k in folds[0] if isinstance(folds[0][k], float)]
    mean = {k: float(np.mean([f[k] for f in folds])) for k in keys}
    mean["routing_drop_A_to_B"] = mean["routing_router"] - mean["routing_router_B"]
    rw = {w: {m: float(np.mean([f["routing_by_workflow"][w][m] for f in folds if w in f["routing_by_workflow"]]))
              for m in ("router", "knn")} for w in WORKFLOWS}
    pred = {
        "P19_router_ge_0.85_and_ge_knn": mean["routing_router"] >= 0.85 and mean["routing_router"] >= mean["routing_knn"],
        "P20_gap_auroc_router_ge_0.75_and_gt_knn": mean["gap_auroc_router"] >= 0.75 and mean["gap_auroc_router"] > mean["gap_auroc_knn"],
        "P21_zero_shot_ge_0.5_and_drop_le_0.05": mean["zero_shot_to_F"] >= 0.5 and mean["routing_drop_A_to_B"] <= 0.05,
        "P22_knn_4shot_gt_zero_shot": mean["knn_4shot_to_F"] > mean["zero_shot_to_F"],
        "P23_subspikes_le_whole_plus_0.01_router_and_oracle": mean["down_router"] <= mean["down_whole"] + 0.01
        and mean["down_oracle"] <= mean["down_whole"] + 0.01,
    }
    res = {"k": K, "shots": SHOTS, "mean": mean, "routing_by_workflow_mean": rw,
           "predictions": {k: bool(v) for k, v in pred.items()}, "folds": folds}
    (RES / "router.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"mean": mean, "routing_by_workflow_mean": rw, "predictions": res["predictions"]}, indent=1))


if __name__ == "__main__":
    main()
