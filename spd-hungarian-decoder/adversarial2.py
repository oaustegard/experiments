"""A4/A5: two more attacks on the reading, run before the writeup.

A4  Rectangular K < N. Eq. 1 of the paper sums over i = 1..N with pi in Pi_N,
    which requires K = N; the method statement says M is N x K. Truncate the
    trained matrix to its first K columns and see what the decoders do.

A5  Is M near rank-1 only because tau was left at 1.0? tau -> 0 sharpens the
    Sinkhorn relaxation toward a permutation matrix, so it is the lever most
    likely to produce a genuinely 2-D M. If a sharper tau breaks the rank-1
    structure, the whole reading is an artifact of my hyperparameter.
"""
import json
from pathlib import Path

import numpy as np
import torch

from decoders import DECODERS, d1_hungarian, rank1_energy, row_argmax_collisions
from run_eval import infer, load, paired_boot, score_rankings

HERE = Path(__file__).resolve().parent
SEED = 20260908


def main():
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(4)
    z = np.load(HERE / "data" / "slates.npz")
    X = torch.from_numpy((z["Xte"] - z["mu"]) / z["sd"])
    Y = z["yte"]; TP = z["teacher_perm_te"].astype(np.int64)
    res = {}

    # A4 --------------------------------------------------------------------
    raw = infer(load("M1", 20), X)
    print("A4  decode-time truncation to K columns (N=50 throughout)")
    print(f"    {'K':>4} {'assigned':>9} {'D1 ndcg10':>10} {'D4':>8} {'D5':>8}")
    a4 = []
    for K in [1, 2, 5, 10, 25, 50]:
        Mk = raw[:, :, :K]
        n_assigned = min(50, K)
        s1 = score_rankings(Y, np.stack([d1_hungarian(M) for M in Mk]), TP)
        s4 = score_rankings(Y, np.stack([DECODERS["D4_expected_pos"](M) for M in Mk]), TP)
        s5 = score_rankings(Y, np.stack([DECODERS["D5_col0"](M) for M in Mk]), TP)
        row = {"K": K, "n_ordinals_the_solver_emits": n_assigned,
               "D1": float(np.nanmean(s1["ndcg10"])),
               "D4": float(np.nanmean(s4["ndcg10"])),
               "D5": float(np.nanmean(s5["ndcg10"]))}
        a4.append(row)
        print(f"    {K:4d} {n_assigned:9d} {row['D1']:10.4f} {row['D4']:8.4f} {row['D5']:8.4f}")
    res["A4_rectangular"] = a4

    # A5 --------------------------------------------------------------------
    print("\nA5  Sinkhorn temperature used in training")
    print(f"    {'tau':>6} {'rank1E':>8} {'collide':>8} {'D1':>8} {'D2':>8} {'D4':>8} {'D5':>8}"
          f"  {'D4-D1 [95% CI]':>28}  {'D5-D1 [95% CI]':>28}")
    a5 = []
    for tag, tau in [("_tau0.25", 0.25), ("_tau0.5", 0.5), ("", 1.0), ("_tau4.0", 4.0)]:
        r = infer(load("M1" + tag, 20), X)
        r1 = float(np.mean([rank1_energy(M) for M in r]))
        coll = float(np.mean([row_argmax_collisions(M)[0] for M in r]))
        s = {k: score_rankings(Y, np.stack([f(M) for M in r]), TP)
             for k, f in DECODERS.items()}
        g4 = paired_boot(s["D4_expected_pos"]["ndcg10"], s["D1_hungarian"]["ndcg10"], rng, 4000)
        g5 = paired_boot(s["D5_col0"]["ndcg10"], s["D1_hungarian"]["ndcg10"], rng, 4000)
        row = {"tau": tau, "rank1_energy": r1, "collisions": coll,
               **{k[:2]: float(np.nanmean(v["ndcg10"])) for k, v in s.items()},
               "D4_minus_D1": g4, "D5_minus_D1": g5}
        a5.append(row)
        print(f"    {tau:6.2f} {r1:8.3f} {coll:8.2f} {row['D1']:8.4f} {row['D2']:8.4f} "
              f"{row['D4']:8.4f} {row['D5']:8.4f}  "
              f"{g4[0]:+.5f} [{g4[1]:+.5f},{g4[2]:+.5f}]  "
              f"{g5[0]:+.5f} [{g5[1]:+.5f},{g5[2]:+.5f}]")
    res["A5_tau"] = a5

    with open(HERE / "out" / "adversarial2.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
