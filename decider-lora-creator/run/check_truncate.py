"""Check web/backends.js truncateRank against numpy's SVD on the same merged delta.

    node run/check_js.mjs && python3 -I run/check_truncate.py

Rebuilds the library merge (mean of three workflow spikes' B A for layer 2's Wi, fp16 from the packed
files) and compares the JS rank-8 singular values and residual with numpy's best rank-8 approximation.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPIKES = ROOT.parent / "adapter-library" / "web" / "models" / "workflows" / "spikes"


def tensor(raw, t):
    return np.frombuffer(raw[t["offset"]:t["offset"] + t["length"]], dtype=np.float16).astype(np.float64).reshape(t["dims"])


def main():
    js = json.loads((ROOT / "out" / "truncate.json").read_text())
    M = 0
    ids = ["agent_trace_observability", "customer_service", "security_incidents"]
    for i in ids:
        sm = json.loads((SPIKES / f"{i}.json").read_text())
        raw = (SPIKES / f"{i}.bin").read_bytes()
        L = sm["layers"][js["layer"]][js["module"]]
        M = M + tensor(raw, L["B"]) @ tensor(raw, L["A"]) / len(ids)
    S = np.linalg.svd(M, compute_uv=False)
    err2 = float((S[8:] ** 2).sum())
    out = {"numpy_sig": S[:8].round(6).tolist(), "js_sig": [round(x, 6) for x in js["sig"]],
           "max_sig_rel_err": float(np.max(np.abs(np.array(js["sig"]) - S[:8]) / S[:8])),
           "numpy_err2": err2, "js_err2": js["err2"], "err2_rel": abs(js["err2"] - err2) / err2,
           "norm2_rel": abs(js["norm2"] - float((S ** 2).sum())) / float((S ** 2).sum())}
    out["ok"] = out["max_sig_rel_err"] < 1e-4 and out["err2_rel"] < 1e-3
    (ROOT / "results" / "check_truncate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
