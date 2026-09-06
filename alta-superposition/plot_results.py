"""Draw the compression curves from `results.json`.

One panel per compiled program: the fraction of held-out inputs the compressed
model still computes exactly, and the largest off-diagonal of the transfer
matrix over live features, both against the code width `d`. Two initialisation
arms per panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = {"code": ("tab:blue", "o", "code SVD"), "data": ("tab:red", "s", "trajectory SVD")}
TITLES = {
    "subleq": "SUBLEQ (looped)",
    "parity_seq": "sequential parity (looped)",
    "parity_ff": "sum-mod-2 parity (feed-forward)",
}


def series(rows, key):
    pairs = [(r["d"], r[key]) for r in rows if r.get(key) is not None]
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    results = json.loads((HERE / "results.json").read_text())
    programs = [p for p in TITLES if any(k.startswith(p + "__") for k in results)]
    fig, axes = plt.subplots(1, len(programs), figsize=(5.2 * len(programs), 4.2))
    axes = [axes] if len(programs) == 1 else list(axes)

    for ax, program in zip(axes, programs):
        twin = ax.twinx()
        live = None
        for arm, (color, marker, label) in ARMS.items():
            record = results.get(f"{program}__{arm}")
            if not record:
                continue
            live = record["n_live"]
            xs, ys = series(record["rows"], "fraction")
            ax.plot(xs, ys, color=color, marker=marker, ms=3, lw=1.4,
                    label=f"exact, {label}")
            xs, ys = series(record["rows"], "transfer_off_diagonal")
            twin.plot(xs, ys, color=color, ls="--", lw=1.1, alpha=0.65,
                      label=f"interference, {label}")
        if live:
            ax.axvline(live, color="0.4", lw=1.0, ls=":")
            ax.annotate(f"live = {live}", xy=(live, 0.5), xytext=(3, 0),
                        textcoords="offset points", fontsize=8, color="0.35",
                        rotation=90, va="center")
        ax.set_title(TITLES[program], fontsize=11)
        ax.set_xlabel("code width $d$")
        ax.set_ylabel("fraction computed exactly")
        ax.set_ylim(-0.05, 1.05)
        twin.set_ylabel("max off-diagonal of $UR^{\\top}$")
        twin.set_ylim(-0.05, 1.05)
        handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
        ax.legend(handles, labels, fontsize=7, loc="center left")

    fig.suptitle(
        "Compressing ALTA's residual stream: behaviour and feature interference",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "curves.png", dpi=160)
    print("wrote", HERE / "curves.png")


if __name__ == "__main__":
    main()
