#!/usr/bin/env python3
"""Per-target numbers and charts for the README, from the swarm state and a final measurement.

    report.py FINAL.json [--out runs/report]

-> report.json (per target: matched/total, mean frac, workers, merges, partial merges, spend,
   $/matched, tail categories, categories that merged), timeline.png (share of functions
   matched and cumulative spend over time, per target, with the run's events marked).
Every results/<commit>.json the gate wrote holds every function (all suites gate every
candidate), so the timeline is measured, not reconstructed."""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "swarm" / "state"
TARGETS = {"neat": "neat* (obscure)", "lua": "Lua 5.4.7 (famous)"}
COLOR = {"neat": "#2a78d6", "lua": "#eb6834"}          # dataviz reference slots 1-2, validated (CVD dE 24.7)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def target(test_id: str) -> str:
    return "lua" if test_id.startswith("lua") else "neat"


def category(reason: str) -> str:
    return (reason or "").split(":", 1)[0].strip() or "?"


def ledger() -> dict:
    last = {}
    for line in (STATE / "ledger.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    return last


def commit_time(sha: str) -> float | None:
    p = subprocess.run(["git", "-C", str(ROOT), "show", "-s", "--format=%ct", sha], capture_output=True, text=True)
    return float(p.stdout.strip()) if p.returncode == 0 and p.stdout.strip() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("final")
    ap.add_argument("--out", default=str(ROOT / "runs" / "report"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    final = json.loads(Path(a.final).read_text())
    led = ledger()
    rep = {}
    for t in TARGETS:
        res = {k: v for k, v in final.items() if target(k) == t}
        recs = [r for r in led.values() if r.get("tests") and target(r["tests"][0]) == t]
        merged = [r for r in recs if r.get("status") == "merged"]
        partial = [r for r in merged if not r.get("gate_fixed") and r.get("partial_gain")]
        spend = sum(r.get("cost_usd") or 0 for r in recs)
        npass = sum(v["status"] == "PASS" for v in res.values())
        tail = collections.Counter(category(v["reason"]) for v in res.values() if v["status"] != "PASS")
        near = sum(1 for v in res.values() if v["status"] != "PASS" and v.get("frac", 0) >= 0.9)
        cleared = collections.Counter(category(re.sub(r"^\w+: ", "", r.get("signature", ""))) for r in merged)
        outcomes = collections.Counter(r.get("status") for r in recs)
        rep[t] = {"label": TARGETS[t], "functions": len(res), "matched": npass,
                  "matched_pct": round(100 * npass / len(res), 1),
                  "mean_frac": round(sum(v.get("frac", 0) for v in res.values()) / len(res), 3),
                  "near_misses_frac_ge_0.9": near,
                  "workers": len(recs), "outcomes": dict(outcomes), "merges": len(merged),
                  "partial_credit_merges": len(partial),
                  "feature_workers": sum(1 for r in recs if r.get("feature")),
                  "rebase_workers": sum(1 for r in recs if r.get("rebase_of")),
                  "spend_usd": round(spend, 2), "usd_per_matched": round(spend / npass, 4) if npass else None,
                  "tail_categories": dict(tail.most_common()), "merged_cluster_categories": dict(cleared.most_common())}
    (out / "report.json").write_text(json.dumps(rep, indent=1))

    # timeline: every gate/refresh measurement, in commit-time order
    pts = []
    for f in (STATE / "results").glob("*.json"):
        ts = commit_time(f.stem)
        if ts is None:
            continue
        r = json.loads(f.read_text())
        if len(r) < 0.9 * len(final):      # a partial measurement (recheck only): skip
            continue
        row = {"time": ts}
        for t in TARGETS:
            vals = [v for k, v in r.items() if target(k) == t]
            row[t] = 100 * sum(v.get("status") == "PASS" for v in vals) / max(1, len(vals))
        pts.append(row)
    pts.sort(key=lambda p: p["time"])
    events = [json.loads(l) for l in (STATE / "events.jsonl").read_text().splitlines() if l.strip()]
    marks = [e for e in events if e.get("kind") == "note"]
    t0 = min([p["time"] for p in pts] + [e["time"] for e in events])
    spend_pts = {t: [] for t in TARGETS}
    for r in sorted(led.values(), key=lambda r: r.get("t_done") or r.get("t_launch") or 0):
        if not r.get("tests"):
            continue
        t = target(r["tests"][0])
        when = r.get("t_done") or r.get("t_launch")
        prev = spend_pts[t][-1][1] if spend_pts[t] else 0.0
        spend_pts[t].append(((when - t0) / 60, prev + (r.get("cost_usd") or 0)))
    rep_tl = {"points": [{**p, "minute": round((p["time"] - t0) / 60, 1)} for p in pts],
              "events": [{"minute": round((e["time"] - t0) / 60, 1), "text": e["text"]} for e in marks]}
    (out / "timeline.json").write_text(json.dumps(rep_tl, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for i, e in enumerate(marks, 1):
            x = (e["time"] - t0) / 60
            ax.axvline(x, color=INK2, linewidth=0.8, linestyle=(0, (2, 3)), zorder=0)
            if ax is ax1:
                ax.text(x + 1, 96, str(i), color=INK2, fontsize=9, va="top")
    finals = {t: [p[t] for p in pts][-1] for t in TARGETS}
    hi = max(finals, key=finals.get)
    for t in TARGETS:
        xs = [(p["time"] - t0) / 60 for p in pts if p["time"] - t0 >= 0]
        ys = [p[t] for p in pts if p["time"] - t0 >= 0]
        ax1.plot(xs, ys, color=COLOR[t], linewidth=2, label=TARGETS[t])
        ax1.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(6, 7 if t == hi else -7),
                     textcoords="offset points", color=INK, fontsize=9, va="center")
        sx, sy = zip(*spend_pts[t]) if spend_pts[t] else ([], [])
        ax2.plot(sx, sy, color=COLOR[t], linewidth=2, drawstyle="steps-post", label=TARGETS[t])
        if sx:
            ax2.annotate(f"${sy[-1]:.2f}", (sx[-1], sy[-1]), xytext=(6, 0), textcoords="offset points",
                         color=INK, fontsize=9, va="center")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("functions matched (%)")
    ax1.set_title("Matching decompilation by a Haiku 5.5 swarm (gcc 13.3 -O2)", loc="left", color=INK,
                  fontsize=12, pad=12)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.9), frameon=False)
    ax2.set_ylabel("worker spend (USD)")
    ax2.set_xlabel("minutes since the first launch (09:46Z, 2026-10-13)")
    fig.text(0.01, 0.005, "\n".join(f"{i}: {e['text'][:110]}".replace("$", r"\$") for i, e in enumerate(marks, 1)),
             fontsize=7.5, color=INK2, va="bottom")
    fig.subplots_adjust(bottom=0.2, right=0.9)
    fig.savefig(out / "timeline.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # progress per dollar: share matched against the target's own cumulative spend (one axis each)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for t in TARGETS:
        sp = spend_pts[t]
        xs, ys = [], []
        for p in pts:
            m = (p["time"] - t0) / 60
            spent = max([s for (mm, s) in sp if mm <= m], default=0.0)
            if spent > 0 or p[t] > 0:
                xs.append(spent)
                ys.append(p[t])
        ax.plot(xs, ys, color=COLOR[t], linewidth=2, label=TARGETS[t])
        if xs:
            ax.annotate(f"{ys[-1]:.0f}% for ${xs[-1]:.2f}".replace("$", r"\$"), (xs[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("worker spend on that target (USD, log scale above \\$1)")
    ax.set_ylabel("functions matched (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Matched functions per dollar, famous vs obscure target", loc="left", color=INK, fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(right=0.82, bottom=0.14)
    fig.savefig(out / "per_dollar.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps({t: {k: v for k, v in r.items() if not isinstance(v, dict)} for t, r in rep.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
