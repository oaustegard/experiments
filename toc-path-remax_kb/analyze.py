"""Paired bootstrap over queries for every arm pair and retrieval mode.

Pairing matters: the same query is scored against every arm, so the difference
per query is the unit to resample. Resampling arms independently would inflate
the interval with between-query variance that the design already removes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N_BOOT = 10_000
SEED = 20260909
KS = (1, 3, 5, 10)
MODES = ("dense", "bm25", "fused")
ARMS = ("A", "B0", "B1", "B2")


def hits_at(ranks: list, k: int) -> np.ndarray:
    return np.array([1.0 if (r is not None and r <= k) else 0.0 for r in ranks])


def mrr(ranks: list) -> np.ndarray:
    return np.array([(1.0 / r) if r is not None else 0.0 for r in ranks])


def paired_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float, float]:
    """Return (mean diff b-a, lo, hi, share of resamples with diff > 0)."""
    d = b - a
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5)), float((boots > 0).mean())


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    rng = np.random.default_rng(SEED)
    out = {"n_queries": res["n_queries"], "n_control": res["n_control"],
           "b2_gold_mean": res["b2_gold_mean"], "absolute": res["arms"],
           "control_absolute": res["control"], "contrasts": {}, "control_contrasts": {}}

    for label, ranks_key, target in (("contrasts", "ranks", "arms"),
                                     ("control_contrasts", "control_ranks", "control")):
        for mode in MODES:
            for base, arm in (("A", "B0"), ("A", "B1"), ("A", "B2"), ("B0", "B1"), ("B1", "B2")):
                ra = res[ranks_key][base][mode]
                rb = res[ranks_key][arm][mode]
                for k in KS:
                    m, lo, hi, p = paired_ci(hits_at(ra, k), hits_at(rb, k), rng)
                    out[label][f"{mode}/{base}->{arm}/R@{k}"] = {
                        "diff": m, "lo": lo, "hi": hi, "p_gt0": p}
                m, lo, hi, p = paired_ci(mrr(ra), mrr(rb), rng)
                out[label][f"{mode}/{base}->{arm}/MRR"] = {
                    "diff": m, "lo": lo, "hi": hi, "p_gt0": p}

    (HERE / "analysis.json").write_text(json.dumps(out, indent=1))

    def row(mode, base, arm, k):
        d = out["contrasts"][f"{mode}/{base}->{arm}/R@{k}"]
        sig = "*" if (d["lo"] > 0 or d["hi"] < 0) else " "
        return f"{d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]{sig}"

    print(f"n = {out['n_queries']} queries\n")
    print("ABSOLUTE (R@1 / R@10)")
    print(f"{'arm':<4} " + "  ".join(f"{m:>16}" for m in MODES))
    for arm in ARMS:
        cells = []
        for m in MODES:
            a = out["absolute"][arm][m]
            cells.append(f"{a['R@1']:.3f} / {a['R@10']:.3f}")
        print(f"{arm:<4} " + "  ".join(f"{c:>16}" for c in cells))

    print("\nPOSITIVE CONTROL — heading path as the query (R@1)")
    print(f"{'arm':<4} " + "  ".join(f"{m:>16}" for m in MODES))
    for arm in ARMS:
        cells = [f"{out['control_absolute'][arm][m]['R@1']:.3f}" for m in MODES]
        print(f"{arm:<4} " + "  ".join(f"{c:>16}" for c in cells))
    cc = out["control_contrasts"]["fused/A->B1/R@1"]
    print(f"  control A->B1 fused R@1: {cc['diff']:+.4f} "
          f"[{cc['lo']:+.4f}, {cc['hi']:+.4f}]")

    print("\nCONTRASTS (diff, 95% paired bootstrap CI; * excludes 0)")
    for mode in MODES:
        print(f"\n  {mode}")
        for base, arm in (("A", "B0"), ("A", "B1"), ("A", "B2"), ("B0", "B1")):
            print(f"    {base}->{arm:<3} " +
                  "  ".join(f"R@{k} {row(mode, base, arm, k)}" for k in (1, 10)))


if __name__ == "__main__":
    main()
