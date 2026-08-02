#!/usr/bin/env python3
"""Plots for the ablation.

Three figures, one per question the issue asks:

  axes.png       recall@10 vs actual bytes/vector, one panel per corpus x
                 metric, remex and HIGGS-like highlighted against the rest of
                 the factorial.  Bytes on the x-axis rather than nominal bits,
                 because the side channels differ between arms.
  marginals.png  the marginal effect of each axis: mean delta recall@10 when
                 flipping A, B or C with the other two held fixed.  This is
                 the plot that answers the issue directly.
  seeds.png      per-seed spread.  Rotation-seed variance is known to produce
                 catastrophic outliers in this family, so min matters as much
                 as mean.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results.json"

REMEX = "haar+exactnorm+scalar"
HIGGS = "rht+blockscale+vector"
BITS = (1, 2, 3, 4, 6, 8)
METRIC_LABEL = {"cosine": "cosine", "ip": "inner product"}


def load():
    return json.loads(RES.read_text())


def _cells(res):
    return [k for k in res if "|" in k]


def fig_axes(res, out):
    cells = _cells(res)
    n = len(cells)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5.2 * ((n + 1) // 2), 8.4),
                             squeeze=False)
    for ax, key in zip(axes.ravel(), cells):
        block = res[key]
        arms = sorted({a for b in BITS if str(b) in block for a in block[str(b)]})
        for arm in arms:
            xs, ys = [], []
            for b in BITS:
                r = block.get(str(b), {}).get(arm)
                if r:
                    xs.append(r["bytes"]["total"])
                    ys.append(r["recall@10"]["mean"])
            if not xs:
                continue
            if arm == REMEX:
                ax.plot(xs, ys, "o-", lw=2.6, ms=7, color="#c1121f",
                        label="remex (haar+exact+scalar)", zorder=5)
            elif arm == HIGGS:
                ax.plot(xs, ys, "s-", lw=2.6, ms=7, color="#0353a4",
                        label="HIGGS-like (rht+block+VQ)", zorder=5)
            elif arm.startswith("control"):
                ax.plot(xs, ys, ":", lw=1.5, color="#6c757d", alpha=0.9,
                        label=arm.replace("control:", "control "))
            else:
                ax.plot(xs, ys, "-", lw=1.0, color="#adb5bd", alpha=0.8,
                        zorder=1)
        ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
        name, metric = key.split("|")
        ax.set_title(f"{name} — {METRIC_LABEL[metric]}")
        ax.set_xscale("log")
        ax.set_xlabel("actual bytes / vector (incl. side channels)")
        ax.set_ylabel("recall@10 vs fp32 exact")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("remex vs the HIGGS lineage — grey lines are the other "
                 "factorial cells", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


AXIS_FLIP = {
    "A rotation: haar -> rht": (0, "haar", "rht"),
    "B norm: exact -> blockscale": (1, "exactnorm", "blockscale"),
    "C codebook: scalar -> vector": (2, "scalar", "vector"),
}


def marginals(res, key, stat="recall@10", field="mean"):
    """Mean delta in `stat` from flipping one axis, other two held fixed."""
    block = res[key]
    out = {a: {} for a in AXIS_FLIP}
    for label, (pos, lo, hi) in AXIS_FLIP.items():
        for b in BITS:
            cell = block.get(str(b))
            if not cell:
                continue
            deltas = []
            for arm in cell:
                parts = arm.split("+")
                if len(parts) != 3 or parts[pos] != lo:
                    continue
                other = parts.copy()
                other[pos] = hi
                twin = "+".join(other)
                if twin in cell:
                    deltas.append(cell[twin][stat][field] - cell[arm][stat][field])
            if deltas:
                out[label][b] = (float(np.mean(deltas)), float(np.std(deltas)))
    return out


def fig_marginals(res, out):
    cells = _cells(res)
    fig, axes = plt.subplots(2, (len(cells) + 1) // 2,
                             figsize=(5.2 * ((len(cells) + 1) // 2), 8.4),
                             squeeze=False)
    colors = {"A rotation: haar -> rht": "#2a9d8f",
              "B norm: exact -> blockscale": "#e9c46a",
              "C codebook: scalar -> vector": "#e76f51"}
    for ax, key in zip(axes.ravel(), cells):
        m = marginals(res, key)
        for label, series in m.items():
            if not series:
                continue
            bs = sorted(series)
            mu = [series[b][0] for b in bs]
            sd = [series[b][1] for b in bs]
            ax.errorbar(bs, mu, yerr=sd, marker="o", capsize=3, lw=2,
                        color=colors[label], label=label)
        ax.axhline(0, color="k", lw=1)
        name, metric = key.split("|")
        ax.set_title(f"{name} — {METRIC_LABEL[metric]}")
        ax.set_xlabel("bits / coordinate")
        ax.set_ylabel("mean $\\Delta$ recall@10 from flipping the axis")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    fig.suptitle("Marginal effect of each axis (positive = the HIGGS side wins)",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_seeds(res, out):
    cells = _cells(res)
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 4.0),
                             squeeze=False)
    for ax, key in zip(axes[0], cells):
        block = res[key]
        for arm, color, lab in ((REMEX, "#c1121f", "remex"),
                                (HIGGS, "#0353a4", "HIGGS-like")):
            bs, mu, lo, hi = [], [], [], []
            for b in BITS:
                r = block.get(str(b), {}).get(arm)
                if r:
                    bs.append(b)
                    mu.append(r["recall@10"]["mean"])
                    lo.append(r["recall@10"]["min"])
                    hi.append(r["recall@10"]["max"])
            if bs:
                ax.plot(bs, mu, "o-", color=color, label=lab)
                ax.fill_between(bs, lo, hi, color=color, alpha=0.2)
        name, metric = key.split("|")
        ax.set_title(f"{name}\n{METRIC_LABEL[metric]}", fontsize=9)
        ax.set_xlabel("bits / coordinate")
        ax.set_ylabel("recall@10")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Rotation-seed spread (band = min..max over 5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    res = load()
    fig_axes(res, HERE / "axes.png")
    fig_marginals(res, HERE / "marginals.png")
    fig_seeds(res, HERE / "seeds.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
