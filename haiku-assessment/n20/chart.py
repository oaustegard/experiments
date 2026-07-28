#!/usr/bin/env python3
"""Generate comparison chart for n=20 results."""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent
data = json.loads((ROOT / "scores.json").read_text())

# Metrics to chart, per task. Each entry: (metric_key, label, "higher_is_better")
# We chart "higher_is_better" metrics so taller = better.
charts = {
    "task1": [
        ("strict_accuracy", "Per-rubric\naccuracy (n=80)", True),
        ("tight_schema_rate", "Tight schema\ncompliance", True),
        # markdown_rate is "lower is better" — invert as "non-markdown rate"
        ("__non_markdown_rate", "No-markdown\ndrift", True),
    ],
    "task2": [
        ("found_mutation_rate", "Found the\nbug", True),
        # added_fix_rate is "lower is better" — invert
        ("__no_fix_section", "No unsolicited\nfix section", True),
        # verbose_rate "lower is better"
        ("__concise_output", "Concise output\n(≤400 chars)", True),
    ],
    "task3": [
        # any_banned_rate "lower is better"
        ("__no_banned_words", "No banned\nhype words", True),
        ("in_length_rate", "Length in\nrange (60-90)", True),
        # any_hallucination_rate "lower is better"
        ("__no_hallucination", "No invented\ndetails", True),
    ],
}

# Pre-compute inverted metrics
def get_metric(summary, key):
    if key == "__non_markdown_rate":
        return 1 - summary["markdown_rate"]
    if key == "__no_fix_section":
        return 1 - summary["added_fix_rate"]
    if key == "__concise_output":
        return 1 - summary["verbose_rate"]
    if key == "__no_banned_words":
        return 1 - summary["any_banned_rate"]
    if key == "__no_hallucination":
        return 1 - summary["any_hallucination_rate"]
    return summary[key]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
task_titles = {
    "task1": "Task 1 — Triage",
    "task2": "Task 2 — Code review",
    "task3": "Task 3 — Voice rewrite",
}

color_v = "#9b9b9b"
color_d = "#3b82f6"

for ax, task in zip(axes, ["task1", "task2", "task3"]):
    metrics = charts[task]
    labels = [m[1] for m in metrics]
    vanilla_vals = [
        get_metric(data[f"{task}_vanilla"]["summary"], m[0])
        for m in metrics
    ]
    downskilled_vals = [
        get_metric(data[f"{task}_downskilled"]["summary"], m[0])
        for m in metrics
    ]
    x = range(len(labels))
    width = 0.38
    b1 = ax.bar([i - width / 2 for i in x], vanilla_vals, width,
                label="Vanilla", color=color_v, edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + width / 2 for i in x], downskilled_vals, width,
                label="Down-skilled", color=color_d, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(task_titles[task], fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(
            f"{h*100:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

# Single legend across the figure
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)

fig.suptitle("Haiku 4.5: vanilla prompt vs. down-skilled prompt — n=20 per cell  (higher = better)",
             fontsize=12, y=1.02)

out = ROOT / "comparison.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
