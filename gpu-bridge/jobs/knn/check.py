"""Compare a fetched knn result against the numpy reference."""
import json, sys
import numpy as np
res = json.load(open(sys.argv[1])); o = res["outputs"]
idx = np.load(o["idx"]["npy"]).reshape(-1, 10); score = np.load(o["score"]["npy"]).reshape(-1, 10)
ri, rs = np.load("ref_idx.npy"), np.load("ref_score.npy")
recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(idx, ri)])
med = lambda a: sorted(a[1:])[len(a[1:]) // 2] if a else None
print(json.dumps({"status": res["status"], "exact_order_match": float((idx == ri).mean()), "recall@10": float(recall),
  "max_score_err": float(np.abs(score - rs).max()), "gpu_median_ms": med(o.get("gpuMs") or []), "wall_median_ms": med(o["wallMs"]),
  "gflop": o["gflop"], "adapter": res.get("adapter"), "logs": res.get("logs")}))
