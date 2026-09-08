"""Scheduled adversarial pass (PLAN.md control 4), run before the writeup.

Four attacks on the headline reading:

A1  Is D4's win an artifact of the softmax temperature it happens to see?
A2  Is M1:D4 (matrix + cheap sort) actually better than M3 (scalar + sort)?
    That decides whether the NxK formulation earns its place independent of
    the Hungarian.
A3  Does the D1-vs-D4 sign survive three training seeds?
A4  Does the picture change when K < N, the rectangular case Eq. 1 does not
    cover?
"""
import json
from pathlib import Path

import numpy as np
import torch

from decoders import DECODERS, d1_hungarian, rank1_energy
from run_eval import infer, load, paired_boot, score_rankings

HERE = Path(__file__).resolve().parent
SEED = 20260908


def d4_temp(M, T):
    z = (M / T); z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return np.argsort(p @ np.arange(M.shape[1], dtype=np.float64), kind="stable")


def main():
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(4)
    z = np.load(HERE / "data" / "slates.npz")
    X = torch.from_numpy((z["Xte"] - z["mu"]) / z["sd"])
    Y = z["yte"]; TP = z["teacher_perm_te"].astype(np.int64)
    res = {}

    raw = infer(load("M1", 20), X)
    d1 = score_rankings(Y, np.stack([d1_hungarian(M) for M in raw]), TP)

    # A1 --------------------------------------------------------------------
    print("A1  D4 softmax temperature sweep (M1, seed 0), NDCG@10 vs D1 = "
          f"{np.nanmean(d1['ndcg10']):.4f}")
    a1 = []
    for T in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
        s = score_rankings(Y, np.stack([d4_temp(M, T) for M in raw]), TP)
        m, lo, hi = paired_boot(s["ndcg10"], d1["ndcg10"], rng, 4000)
        a1.append({"T": T, "ndcg10": float(np.nanmean(s["ndcg10"])),
                   "vs_D1": m, "ci": [lo, hi]})
        print(f"    T={T:6.2f}  ndcg10 {np.nanmean(s['ndcg10']):.4f}   "
              f"D4-D1 {m:+.5f} [{lo:+.5f},{hi:+.5f}]"
              + ("  *" if not (lo <= 0 <= hi) else ""))
    res["A1_temperature"] = a1

    # A2 --------------------------------------------------------------------
    print("\nA2  does the NxK matrix beat a scalar score, once the Hungarian is dropped?")
    d4 = score_rankings(Y, np.stack([DECODERS["D4_expected_pos"](M) for M in raw]), TP)
    m3 = score_rankings(Y, np.argsort(-infer(load("M3", 20), X), axis=1, kind="stable"), TP)
    a2 = {}
    for met in ("ndcg1", "ndcg10", "ndcg50", "recall10", "tau"):
        a2[met] = paired_boot(d4[met], m3[met], rng)
        m, lo, hi = a2[met]
        print(f"    M1:D4 - M3:sort [{met:8}] {m:+.5f} [{lo:+.5f},{hi:+.5f}]"
              + ("  *" if not (lo <= 0 <= hi) else ""))
    res["A2_matrix_vs_scalar"] = a2

    # A3 --------------------------------------------------------------------
    print("\nA3  three training seeds")
    a3 = []
    for tag in ["", "_s1", "_s2"]:
        r = infer(load("M1" + tag, 20), X)
        s1 = score_rankings(Y, np.stack([d1_hungarian(M) for M in r]), TP)
        s4 = score_rankings(Y, np.stack([DECODERS["D4_expected_pos"](M) for M in r]), TP)
        s5 = score_rankings(Y, np.stack([DECODERS["D5_col0"](M) for M in r]), TP)
        s2 = score_rankings(Y, np.stack([DECODERS["D2_row_greedy"](M) for M in r]), TP)
        m3s = score_rankings(Y, np.argsort(-infer(load("M3" + tag, 20), X), axis=1,
                                           kind="stable"), TP)
        r1 = float(np.mean([rank1_energy(M) for M in r]))
        row = {"seed": tag or "_s0", "rank1_energy": r1,
               "D1": float(np.nanmean(s1["ndcg10"])), "D2": float(np.nanmean(s2["ndcg10"])),
               "D4": float(np.nanmean(s4["ndcg10"])), "D5": float(np.nanmean(s5["ndcg10"])),
               "M3": float(np.nanmean(m3s["ndcg10"])),
               "D4_minus_D1": paired_boot(s4["ndcg10"], s1["ndcg10"], rng, 4000),
               "D1_minus_D2": paired_boot(s1["ndcg10"], s2["ndcg10"], rng, 4000)}
        a3.append(row)
        print(f"    {row['seed']}  rank1E {r1:.3f}  D1 {row['D1']:.4f}  D2 {row['D2']:.4f}  "
              f"D4 {row['D4']:.4f}  D5 {row['D5']:.4f}  M3 {row['M3']:.4f}   "
              f"D4-D1 {row['D4_minus_D1'][0]:+.5f} "
              f"[{row['D4_minus_D1'][1]:+.5f},{row['D4_minus_D1'][2]:+.5f}]")
    res["A3_seeds"] = a3

    with open(HERE / "out" / "adversarial.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
