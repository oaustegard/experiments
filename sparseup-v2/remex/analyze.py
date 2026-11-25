"""Macro-average every arm across the four datasets with a bootstrap CI, and list the macro frontier.

Macro = unweighted mean of the four dataset means. CI: 1,000 resamples, each resampling
queries independently within every dataset (paired against fp32 full k), then averaging.
Writes results/summary.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DS = ["scifact", "nfcorpus", "fiqa", "arguana"]
N_BOOT = 1000


def pareto(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    return np.array([not np.any((xs <= xs[i]) & (ys >= ys[i]) & ((xs < xs[i]) | (ys > ys[i])))
                     for i in range(len(xs))])


def main():
    pq, arms = {}, {}
    for d in DS:
        z = np.load(HERE / "cache" / f"perquery_{d}.npz")
        pq[d] = dict(zip(z["keys"].tolist(), z["vals"]))
        a = json.loads((HERE / "results" / f"arms_{d}.json").read_text())
        arms[d] = {f"{r['k']}|{r['codebook']}|{r['bits']}|{r['scale']}|{r.get('query', 'fp32')}": r for r in a["arms"]}
    rng = np.random.default_rng(0)
    idx = {d: rng.integers(0, len(pq[d]["base_ndcg"]), size=(N_BOOT, len(pq[d]["base_ndcg"]))) for d in DS}
    keys = [k for k in arms[DS[0]] if all(k in arms[d] for d in DS)]
    rows = []
    for key in keys:
        deltas = {d: pq[d][key] - pq[d]["base_ndcg"] for d in DS}
        boot = np.mean([deltas[d][idx[d]].mean(1) for d in DS], axis=0)
        r0 = arms[DS[0]][key]
        rows.append({
            "key": key, "k": r0["k"], "codebook": r0["codebook"], "bits": r0["bits"], "scale": r0["scale"],
            "query": r0.get("query", "fp32"),
            "macro_d_ndcg10": float(np.mean([deltas[d].mean() for d in DS])),
            "macro_d_ndcg10_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "macro_d_r100": float(np.mean([arms[d][key]["d_r100"] for d in DS])),
            "macro_bytes_fixed": float(np.mean([arms[d][key]["bytes_fixed"] for d in DS])),
            "macro_bytes_varint": float(np.mean([arms[d][key]["bytes_varint"] for d in DS])),
            "macro_nnz": float(np.mean([arms[d][key]["nnz"] for d in DS])),
            "per_dataset_d_ndcg10": {d: arms[d][key]["d_ndcg10"] for d in DS},
            "per_dataset_ci": {d: arms[d][key]["d_ndcg10_ci"] for d in DS},
        })
    doc_rows = [r for r in rows if r["query"] == "fp32"]
    for xkey in ["macro_bytes_fixed", "macro_bytes_varint"]:
        m = pareto([r[xkey] for r in doc_rows], [r["macro_d_ndcg10"] for r in doc_rows])
        for r, f in zip(doc_rows, m):
            r["frontier_" + xkey.split("_")[-1]] = bool(f)
    base = {d: {"ndcg10": float(pq[d]["base_ndcg"].mean()),
                "r100": json.loads((HERE / "results" / f"arms_{d}.json").read_text())["base_r100"],
                "n_queries": len(pq[d]["base_ndcg"]),
                "n_docs": json.loads((HERE / "results" / f"arms_{d}.json").read_text())["n_docs"]} for d in DS}
    by_key = {r["key"].replace("|", "_"): {"d": r["macro_d_ndcg10"], "lo": r["macro_d_ndcg10_ci"][0],
                                            "hi": r["macro_d_ndcg10_ci"][1], "bytes_fixed": r["macro_bytes_fixed"],
                                            "bytes_varint": r["macro_bytes_varint"], "nnz": r["macro_nnz"],
                                            "varint_bytes_per_term": r["macro_bytes_varint"] / max(r["macro_nnz"], 1e-9),
                                            **{f"d_{d}": v for d, v in r["per_dataset_d_ndcg10"].items()}}
              for r in rows}
    out = {"baseline": base, "macro_base_ndcg10": float(np.mean([base[d]["ndcg10"] for d in DS])),
           "by_key": by_key, "arms": rows}
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    print("baseline", {d: round(base[d]["ndcg10"], 4) for d in DS}, "macro", round(out["macro_base_ndcg10"], 4))
    show = sorted(doc_rows, key=lambda r: r["macro_bytes_fixed"])
    for r in show:
        if r["frontier_fixed"] or r["codebook"] in ("fp32", "fp16", "table8", "anserini8") or \
                (r["k"] is None and r["scale"] == "none"):
            ci = r["macro_d_ndcg10_ci"]
            print(f"{r['key']:34s} B {r['macro_bytes_fixed']:7.1f}/{r['macro_bytes_varint']:7.1f} "
                  f"d {r['macro_d_ndcg10']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]{' *F' if r['frontier_fixed'] else ''}")
    print("query-side arms:")
    for r in rows:
        if r["query"] != "fp32":
            ci = r["macro_d_ndcg10_ci"]
            print(f"{r['key']:34s} d {r['macro_d_ndcg10']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
