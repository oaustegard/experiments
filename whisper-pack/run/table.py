"""Table of WER per scheme with the paired-bootstrap difference against f32, pooled over sets.

usage: table.py results/sweep-<model>.jsonl [SET,SET...]   (default: ls-clean,ls-other pooled)
Writes results/table-<model>.json next to the jsonl.
"""
import json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
sets = (sys.argv[2] if len(sys.argv) > 2 else "ls-clean,ls-other").split(",")
recs = {}
for line in path.open():
    r = json.loads(line)
    recs[(r["scheme"], r["set"])] = r
schemes = list(dict.fromkeys(s for s, _ in recs))
rng = np.random.RandomState(0)


def pooled(s):
    e = np.concatenate([recs[(s, t)]["utt_errors"] for t in sets]).astype(float)
    w = np.concatenate([recs[(s, t)]["utt_ref_words"] for t in sets]).astype(float)
    return e, w


e0, w0 = pooled("f32")
idx = rng.randint(0, len(w0), size=(10000, len(w0)))
rows = []
for s in schemes:
    if any((s, t) not in recs for t in sets):
        continue
    e, w = pooled(s)
    d = (e[idx].sum(1) - e0[idx].sum(1)) / w[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    q = recs[(s, sets[0])]["quant"]
    row = {"scheme": s, "raw_mb": round(q["raw_bytes"] / 1e6, 1), "entropy_mb": round(q["entropy_bytes"] / 1e6, 1),
           **{t: round(100 * recs[(s, t)]["wer"], 2) for t in sets},
           "pooled": round(100 * e.sum() / w.sum(), 2), "diff": round(100 * (e.sum() - e0.sum()) / w.sum(), 2),
           "ci": [round(100 * lo, 2), round(100 * hi, 2)]}
    rows.append(row)
    print(f"{s:44s} {row['raw_mb']:7.1f} {row['entropy_mb']:7.1f} " + " ".join(f"{row[t]:6.2f}" for t in sets)
          + f"  pooled {row['pooled']:6.2f}  {row['diff']:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]")
out = path.with_name(path.name.replace("sweep-", "table-").replace(".jsonl", "-" + "+".join(sets) + ".json"))
out.write_text(json.dumps({"sets": sets, "rows": rows}, indent=1))
