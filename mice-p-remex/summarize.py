"""One table over every scored arm: bytes/token, nDCG@10, paired delta vs fp32
(95% bootstrap CI), Spearman rho and top-10 overlap against the fp32 ranking.
Pools all datasets' queries for the 'pooled' delta (each query weighted once).

    python3 summarize.py   -> prints the table, writes results/summary.json
"""
import json
from pathlib import Path

import evaluate as E
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ["scifact", "nfcorpus", "fiqa", "scidocs"]
ARMS = ["fp32", "plaid-2", "plaid-4", "plaid-8", "plaid-n-2", "plaid-n-4", "plaid-rx-1", "plaid-rx-2",
        "remex-c-2", "remex-2"]
PAIRS = [("plaid-rx-2", "plaid-2"), ("plaid-rx-2", "plaid-4"), ("plaid-n-2", "plaid-2"),
         ("plaid-rx-2", "plaid-n-2"), ("plaid-rx-2", "remex-c-2"), ("plaid-rx-1", "plaid-4"),
         ("remex-c-2", "plaid-4")]


def main():
    out, pooled = {}, {}
    for ds in DATASETS:
        sdir = HERE / "cache" / "scores" / ds
        base_path = sdir / "m130.fp32.scores.json"
        if not base_path.exists():
            continue
        base = json.load(open(base_path))
        pq_base = E.per_query(ds, base)
        arms_meta = json.loads((HERE / "results" / f"arms_{ds}.json").read_text()) \
            if (HERE / "results" / f"arms_{ds}.json").exists() else {}
        for arm in ARMS:
            p = sdir / f"m130.{arm}.scores.json"
            if not p.exists():
                continue
            s = json.load(open(p))
            if set(s) != set(base):
                continue  # partial file
            pq = E.per_query(ds, s)
            m, lo, hi, n = E.paired(pq, pq_base)
            rho, ov = E.fidelity(s, base)
            out.setdefault(ds, {})[arm] = {
                "ndcg10": round(float(np.mean(list(pq.values()))), 4), "delta": round(m, 4),
                "ci": [round(lo, 4), round(hi, 4)], "n": n, "spearman": round(rho, 4), "top10": round(ov, 3),
                "bytes_per_token": arms_meta.get(arm, {}).get("bytes_per_token", 512.0 if arm == "fp32" else None),
                "err_up": arms_meta.get(arm, {}).get("err_up")}
            pooled.setdefault(arm, {}).update({f"{ds}/{q}": v for q, v in pq.items()})
    for arm in ARMS:
        if arm in pooled and len(pooled[arm]) == len(pooled["fp32"]):
            m, lo, hi, n = E.paired(pooled[arm], pooled["fp32"])
            out.setdefault("pooled", {})[arm] = {"delta": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n": n,
                                                 "mean_delta_over_datasets": round(float(np.mean(
                                                     [out[d][arm]["delta"] for d in out if d != "pooled"])), 4)}
    pairs = {}
    for a, b in PAIRS:
        if a in pooled and b in pooled:
            m, lo, hi, n = E.paired(pooled[a], pooled[b])
            pairs[f"{a}__{b}"] = {"delta": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n": n}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / "paired.json").write_text(json.dumps(pairs, indent=1))
    (HERE / "results" / "perquery.json").write_text(json.dumps(
        {a: {k: round(v, 6) for k, v in pq.items()} for a, pq in pooled.items()}))
    for k, r in pairs.items():
        print(f"{k:24s} {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] n={r['n']}")
    for ds, rows in out.items():
        print(f"\n{ds}")
        for arm, r in rows.items():
            if ds == "pooled":
                print(f"  {arm:11s} delta {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] n={r['n']}  "
                      f"mean-over-datasets {r['mean_delta_over_datasets']:+.4f}")
            else:
                b = r["bytes_per_token"]
                print(f"  {arm:11s} B/tok {b if b is not None else '?':>5}  nDCG {r['ndcg10']:.4f}  delta {r['delta']:+.4f} "
                      f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  rho {r['spearman']:.4f}  top10 {r['top10']:.3f}")


if __name__ == "__main__":
    main()
