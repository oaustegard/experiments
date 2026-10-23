"""Paired bootstrap CIs between r6 builds (same queries). -> results/r6_pairs.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, LS, WORK  # noqa: E402

sys.path.insert(0, str(LS / "run"))
from img_common import eval_queries, is_test, rows  # noqa: E402


def ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [round(float(d.mean()), 4), round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def main():
    _, target = eval_queries()
    cid = {r["i"]: r["cid"] for r in rows()}
    test = np.array([is_test(cid[t]) for t in target])
    L = lambda n: np.load(WORK / (f"{n}.npz" if n.startswith("packed_") else f"r6_{n}.npz"))  # noqa: E731
    pairs = {"v64k_mix10_vs_trim_int3": ("vc_v64k_rhq_mix10", "vc_trim_int3_g32"),
             "v64k_rhq3_vs_trim_int3": ("vc_v64k_rhq3", "vc_trim_int3_g32"),
             "trim_rhq3_vs_trim_int3": ("vc_trim_rhq3", "vc_trim_int3_g32"),
             "v64k_float_vs_trim_float": ("vc_v64k_float", "vc_trim_float"),
             "packed_v64k_rhq_vs_trim_t3w": ("packed_v64k_rhq", "packed_trim_t3w"),
             "packed_trim_rhq3_vs_trim_t3w": ("packed_trim_rhq3", "packed_trim_t3w")}
    out = {}
    for name, (a, b) in pairs.items():
        A, Bq = L(a), L(b)
        out[name] = {k: ci(A[k] - Bq[k]) for k in ("real_overlap", "synth_overlap", "synth_hit10")}
        out[name].update({f"img_test_{k}": ci((A[f"img_{k}"] - Bq[f"img_{k}"])[test]) for k in ("overlap", "hit10")})
        print(name, out[name])
    (HERE / "results" / "r6_pairs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
