"""Growth plot: D(t) (best exact value over computed n) vs sqrt(t), t, 2t-1.

Reads bf_results.json; writes growth.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    with open("bf_results.json") as f:
        rows = json.load(f)
    by_t: dict[int, int] = {}
    for r in rows:
        by_t[r["t"]] = max(by_t.get(r["t"], 0), r["D"])
    ts = sorted(by_t)
    vals = [by_t[t] for t in ts]

    tt = np.linspace(1, max(ts) + 0.5, 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tt, np.sqrt(tt), "--", label=r"$\sqrt{t}$ (conjecture rate)")
    ax.plot(tt, tt, ":", label=r"$t$")
    ax.plot(tt, 2 * tt - 2, "-.", label=r"$2t-2$ (Beck–Fiala)")
    ax.plot(tt, 2 * tt - 3, ":", alpha=0.6,
            label=r"$2t-3$ (Bednarchak–Helm, $t\geq 3$)")
    ax.plot(ts, vals, "o-", color="black", lw=2,
            label=r"exact $D(t)$, $n$ searched (this work)")
    for t, v in zip(ts, vals):
        ax.annotate(f"D({t})≥{v}", (t, v), textcoords="offset points",
                    xytext=(8, -12))
    ax.set_xlabel("max element degree t")
    ax.set_ylabel("discrepancy")
    ax.set_title("Beck–Fiala small-t exact values vs growth curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("growth.png", dpi=150)
    print("wrote growth.png:", {t: v for t, v in zip(ts, vals)})


if __name__ == "__main__":
    main()
