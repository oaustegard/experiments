"""Two controls on the measurement itself, run before any trained-M result.

POSITIVE: build an M whose optimal assignment provably recovers a known true
ranking, and where row-greedy does not. If the harness cannot show D1 > D2
here, a null on a trained M means nothing.

NEGATIVE: build a rank-1 M and confirm D1 is exactly a sort, so the G3
diagnostic in PLAN.md has the interpretation it claims.
"""
import json
from pathlib import Path

import numpy as np

from decoders import DECODERS, ndcg, rank1_energy, row_argmax_collisions

HERE = Path(__file__).resolve().parent
N = 50
SEED = 20260908


def positive_control(sigma, n_slates=1000, N=N, rng=None):
    """M[i,j] = -|i-j| + sigma*noise. True ranking is the identity."""
    rng = rng or np.random.default_rng(SEED)
    # graded labels decreasing with true rank, MSLR-like mix
    labels = np.zeros(N, dtype=np.float32)
    labels[:2], labels[2:6], labels[6:16], labels[16:30] = 4, 3, 2, 1
    base = -np.abs(np.subtract.outer(np.arange(N), np.arange(N))).astype(np.float64)
    out = {k: [] for k in DECODERS}
    coll, r1 = [], []
    for _ in range(n_slates):
        M = base + sigma * rng.normal(size=(N, N))
        coll.append(row_argmax_collisions(M)[0])
        r1.append(rank1_energy(M))
        for k, f in DECODERS.items():
            out[k].append(ndcg(labels, f(M), 10))
    return ({k: float(np.mean(v)) for k, v in out.items()},
            float(np.mean(coll)), float(np.mean(r1)))


def negative_control(n_slates=500, N=N, rng=None):
    """Rank-1 M with a decreasing position factor: D1 must equal sort-by-a."""
    rng = rng or np.random.default_rng(SEED + 1)
    agree = 0
    for _ in range(n_slates):
        a = rng.normal(size=N)
        b = np.sort(rng.uniform(0.5, 2.0, size=N))[::-1]  # decreasing in position
        M = np.outer(a, b)
        agree += int(np.array_equal(DECODERS["D1_hungarian"](M), np.argsort(-a, kind="stable")))
    return agree / n_slates


def main():
    rows = []
    print(f"{'sigma':>7} {'collide':>8} {'rank1E':>7}  " +
          "  ".join(f"{k:>15}" for k in DECODERS))
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        sc, coll, r1 = positive_control(sigma)
        rows.append({"sigma": sigma, "collisions": coll, "rank1_energy": r1, **sc})
        print(f"{sigma:7.1f} {coll:8.2f} {r1:7.3f}  " +
              "  ".join(f"{sc[k]:15.4f}" for k in DECODERS))

    agree = negative_control()
    print(f"\nnegative control: rank-1 M, D1 == sort by row factor in "
          f"{agree*100:.1f}% of 500 draws")
    (HERE / "out").mkdir(exist_ok=True)
    with open(HERE / "out" / "controls.json", "w") as fh:
        json.dump({"positive": rows, "negative_rank1_agreement": agree}, fh, indent=2)


if __name__ == "__main__":
    main()
