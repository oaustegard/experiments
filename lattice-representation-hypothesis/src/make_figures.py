"""Figures for the Lattice Representation Hypothesis experiment.

Reads whatever result JSONs exist and skips the panels it cannot fill, so
this is safe to run before every arm has landed.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"
FIG = HERE / "figures"

# Colour-blind-safe, readable on light and dark backgrounds.
MEET = "#0F9D58"
JOIN = "#D93025"
NEUTRAL = "#5F6368"
ACCENT = "#1A73E8"

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_sweeps(a):
    """The three synthetic sweeps, plus the meet line pinned at zero."""
    s = a["summary"]
    dim, coh, den = (s["dimension_sweep_series"], s["coherence_sweep_series"],
                     s["density_sweep_series"])

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # Two normalizations, because they disagree in SIGN here and reporting
    # only the first would say "low dimension is benign", which is wrong.
    # ratio = phantoms / |closure|; rate = phantoms / objects outside the union.
    ax[0].plot(dim["d"], dim["join_overshoot_mean_all"], "o-", color=JOIN,
               label="phantoms / |join|")
    ax[0].plot(dim["d"], dim["phantom_rate_mean_all"], "^--", color="#8E24AA",
               label="phantoms / objects at risk")
    ax[0].axhline(0, color=MEET, lw=2, label="meet overshoot (exactly 0)")
    ax[0].set_xscale("log", base=2)
    ax[0].invert_xaxis()
    ax[0].set_xlabel("embedding dimension $d$  (12 attributes)")
    ax[0].set_ylabel("mean overshoot")
    ax[0].set_title("Squeezing dimensions dissolves the lattice;\n"
                    "per object at risk the join gets worse", fontsize=9)
    ax[0].legend(fontsize=6.5, loc="upper left")

    ax[1].plot(coh["coherence"], coh["join_overshoot_mean_all"], "o-", color=JOIN)
    ax[1].axhline(0, color=MEET, lw=2)
    ax[1].set_xlabel("mutual coherence of attribute directions")
    ax[1].set_title("Correlated directions shrink the gap\n($d=64$, full rank throughout)",
                    fontsize=9)

    ax[2].plot(den["p"], den["join_overshoot_mean_all"], "o-", color=JOIN)
    ax[2].axhline(0, color=MEET, lw=2)
    ax[2].set_xlabel("attribute density $p$")
    ax[2].set_title("Sparse contexts are the worst case", fontsize=9)

    for a_ in ax:
        a_.set_ylim(-0.04, 0.95)
    fig.tight_layout()
    fig.savefig(FIG / "sweeps.png", bbox_inches="tight")
    plt.close(fig)


def fig_distribution(a):
    """Where the overshoot mass actually sits, at the baseline config."""
    b = a["experiment_1_baseline"]
    labels = b["hist_labels"]
    hist = b["pooled_overshoot_hist_all"]
    total = sum(hist) or 1

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(range(len(hist)), [h / total for h in hist], color=JOIN, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("join overshoot (fraction of the join that is phantom)")
    ax.set_ylabel("fraction of concept pairs")
    ax.set_title("Join overshoot distribution, $d=64$, 12 attributes, $p=0.5$\n"
                 "meet is a single bar at zero", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "overshoot_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_lattice_membership(dd):
    """How often the paper's union-join is an element of its own lattice."""
    rows = dd["rows"]
    rows = sorted(rows, key=lambda r: r["union_join_NOT_lattice_element_frac"])
    labels = [r["label"] for r in rows]
    vals = [r["union_join_NOT_lattice_element_frac"] for r in rows]
    over = [r["overshoot_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(rows) + 1.6))
    y = range(len(rows))
    ax.barh(list(y), vals, color=JOIN, alpha=0.85, label="plain union is not closed")
    ax.plot(over, list(y), "o", color=ACCENT, ms=5, label="mean overshoot of the closure-join")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of concept pairs")
    ax.set_title("How far the lattice join sits from plain disjunction\n(a property of the context, NOT an error rate)", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "union_join_not_a_lattice_element.png", bbox_inches="tight")
    plt.close(fig)


def fig_noise_reversal(b):
    """The result that reversed the thesis: under probe error the closure-join
    degrades more slowly than the bare-intersection meet."""
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
    for k, (name, ctx) in enumerate(b["contexts"].items()):
        rows = ctx["noise_sweep"]
        p_ = [r["flip_rate"] for r in rows]
        ax[k].plot(p_, [r["meet_jaccard"]["mean"] for r in rows], "o-", color=MEET,
                   label="meet  ($A_1 \\cap A_2$)")
        ax[k].plot(p_, [r["join_jaccard"]["mean"] for r in rows], "s-", color=JOIN,
                   label="join  $(A_1 \\cup A_2)\'\'$")
        ax[k].plot(p_, [r["union_jaccard"]["mean"] for r in rows], "^--", color=NEUTRAL,
                   lw=1, ms=4, label="plain union (not a concept)")
        ax[k].set_xlabel("per-cell probe error rate")
        ax[k].set_title(f"WordNet context: {name}", fontsize=9)
    ax[0].set_ylabel("Jaccard vs. true lattice")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("The closure absorbs probe noise; the bare intersection does not",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "noise_reversal.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    a = load("arm_a_synthetic.json")
    if a:
        fig_sweeps(a); made.append("sweeps.png")
        fig_distribution(a); made.append("overshoot_distribution.png")

    b = load("arm_b_wordnet.json")
    if b:
        fig_noise_reversal(b); made.append("noise_reversal.png")

    dd = load("arm_d_join_operators.json")
    if dd:
        fig_lattice_membership(dd)
        made.append("union_join_not_a_lattice_element.png")

    print("wrote:", ", ".join(made) if made else "nothing (no result files yet)")


if __name__ == "__main__":
    main()
