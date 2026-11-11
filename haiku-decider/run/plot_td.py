"""Place the Haiku arms on the Typed Decisions zero-shot leaderboard.

    python3 run/plot_td.py [--out DIR]   # -> leaderboard.png (every general zero-shot row) and hero.png (the top rows)

Rows come from the dataset card's first table (general models, zero-shot), read from
browser-decider/data/td/README.md; the Haiku rows come from results/td/summary.json (the shipped,
calibrated arms). Accuracy is agreement with the gold's top option on the 2,000 test decisions.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

PAPER, INK, MUTED, RULE = "#f5f0e8", "#1e2a4a", "#8b8577", "#d5cfc3"
INDIGO, CORAL = "#2b3a67", "#cf6853"


def card_rows():
    text = (experiment("browser-decider") / "data" / "td" / "README.md").read_text()
    table = text.split("## Leaderboard", 1)[1].split("### Fitted", 1)[0]
    rows = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.match(r"^(\d+|–)$", cells[0]):
            continue
        name = re.sub(r"\*\*|\[([^\]]+)\]\([^)]*\)", lambda m: m.group(1) or "", cells[1])
        name = re.sub(r"\s*\((#\d+)\)", "", name)
        name = re.sub(r"\s+via .*$", "", name)
        name = re.sub(r"\s*\(`[^`]*`\)", "", name).strip()
        acc = float(re.sub(r"[^\d.]", "", cells[3]))
        rows.append((name, acc))
    return rows


def plot(rows, path, title, min_acc=None):
    rows = sorted(rows, key=lambda r: r[1])
    if min_acc is not None:
        rows = [r for r in rows if r[1] >= min_acc or "Haiku" in r[0] or "Jev 1.13" in r[0]]
    h = 0.26 * len(rows) + 1.2 if min_acc is None else 5.3  # the hero keeps a link-preview shape
    fig, ax = plt.subplots(figsize=(10, h), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for i, (name, acc) in enumerate(rows):
        if "Haiku" in name:
            c, w, s = CORAL, "bold", 70
        elif "Jev 1.13" in name:
            c, w, s = INDIGO, "bold", 70
        elif "jev" in name.lower():
            c, w, s = INDIGO, "normal", 34
        else:
            c, w, s = MUTED, "normal", 26
        ax.hlines(i, 0.3, acc, color=RULE, lw=0.8, zorder=1)
        ax.scatter([acc], [i], s=s, color=c, zorder=3)
        label = f"{acc:.4f}" if round(acc, 3) != acc else f"{acc:.3f}"
        if 0.725 < acc < 0.735:  # keep the label clear of the dashed reference line
            ax.text(acc - 0.003, i, label, va="center", ha="right", fontsize=7, color=c, fontweight=w)
        else:
            ax.text(acc + 0.003, i, label, va="center", fontsize=7, color=c, fontweight=w)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    for lbl, (name, _) in zip(ax.get_yticklabels(), rows):
        if "Haiku" in name:
            lbl.set_color(CORAL)
            lbl.set_fontweight("bold")
        elif "Jev 1.13" in name:
            lbl.set_color(INDIGO)
            lbl.set_fontweight("bold")
        else:
            lbl.set_color(INK)
    lo = min(r[1] for r in rows) - 0.03
    ax.set_xlim(max(0.3, lo), 0.81)
    ax.set_ylim(-0.7, len(rows) + 0.4)
    ax.axvline(0.735, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.text(0.736, len(rows) - 0.1, "teacher agrees with itself: 0.735", fontsize=7, color=INK, va="center")
    ax.set_xlabel("accuracy: agreement with the gold's top option, 2,000 test decisions", fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="x", colors=INK, labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "td"))
    a = ap.parse_args()
    s = json.loads((HERE / "results" / "td" / "summary.json").read_text())
    rows = [r for r in card_rows() if not r[0].startswith(("Prior", "Uniform"))]
    rows += [("Claude Haiku 5.5, one call (this skill)", round(s["arms"]["one"]["accuracy"], 4)),
             ("Claude Haiku 5.5, three-call stack (this skill)", round(s["arms"]["stack"]["accuracy"], 4))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot(rows, out / "leaderboard.png", "Typed Decisions, general models scored zero-shot")
    plot(rows, out / "hero.png", "Typed Decisions, zero-shot: the top of the table, Jev and Haiku", min_acc=0.735)
    print(len(rows), "rows ->", out)


if __name__ == "__main__":
    main()
