"""Curves, tables and plots from results/runs/*/{summary.json,events.jsonl}.

A run's curve is the best score among programs evaluated so far, as a function
of dollars spent when each evaluation finished (and, separately, of wall-clock
minutes). Only evaluations inside the run's wall-clock ceiling count. The
initial program is scored at $0.

Writes results/summary.json, results/table.md, results/curves.png and, when
the tuning runs exist, results/tuning.png.
"""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# HSP_RESULTS lets an interim look write its tables elsewhere; runs are always read from results/runs.
R = HERE / "results"
OUT = Path(__import__("os").environ.get("HSP_RESULTS_OUT", str(R)))
EPS = 1e-6   # a score difference below this is a tie (evaluator tolerance)
WALL = {"A": 3600, "Am": 3600, "Al": 3600, "B": 3600, "C": 3600, "L": 3600, "T": 1800}
ARMS = {"A": "A  Sonnet loop (OpenEvolve), effort high", "Am": "Am Sonnet loop, effort medium",
        "Al": "Al Sonnet loop, effort low", "B": "B  Haiku swarm", "C": "C  Haiku proposers, Sonnet writer",
        "L": "L  Haiku swarm, all effort low"}
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "L": "#eda100", "Am": "#e87ba4", "Al": "#6250d6",
          "T0": "#2a78d6", "T1": "#eb6834", "T2": "#1baf7a", "T3": "#eda100"}
PROBLEM_NAMES = {"circle": "Circle packing, n=26 (1.0 = best known, 2.635983)",
                 "heilbronn": "Heilbronn triangle, n=11 (1.0 = AlphaEvolve's construction)",
                 "hexagon": "Hexagon packing, n=11 (1.0 = AlphaEvolve's construction)"}
CHECK_SPEND = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
THRESHOLDS = [0.99, 0.999, 0.9999]


def load():
    runs = {}
    for d in sorted((R / "runs").glob("*")):
        if not (d / "summary.json").exists() or not (d / "events.jsonl").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        arm = s.get("arm") or d.name[0]
        problem = s.get("problem", "circle")
        ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
        ev = [e for e in ev if e.get("kind") == "eval" and e["te"] <= WALL[arm]]
        if problem == "hexagon":
            # Scores the stock evaluator gave to geometrically impossible packings
            # (see common.score_of) count as invalid.
            from common import HEX_MAX_SCORE
            for e in ev:
                if e["score"] > HEX_MAX_SCORE:
                    e["score"], e["invalid_accepted"] = 0.0, True
        ev.sort(key=lambda e: e["spent"])
        runs[d.name] = {"arm": arm, "problem": problem, "summary": s, "events": ev}
    return runs


def best_at(ev, x, key="spent"):
    vals = [e["score"] for e in ev if e[key] <= x]
    return max(vals) if vals else float("nan")


def first_reach(ev, thr, key="spent"):
    hits = [e[key] for e in ev if e["score"] >= thr]
    return min(hits) if hits else None


def final_x(run, key="spent"):
    s = run["summary"]
    if key == "spent":
        return s["spent"]
    if key == "spent_api":
        return s.get("spent_api", s["spent"])
    return min(s.get("elapsed", WALL[run["arm"]]), WALL[run["arm"]])


def per_run(run):
    ev, s = run["events"], run["summary"]
    spent = s["spent"]
    progs = len({e["sha"] for e in ev})
    return {
        "run": s["run"], "arm": run["arm"], "seed": s.get("seed"), "spent": round(spent, 4),
        "spent_api": round(s.get("spent_api", float("nan")), 4), "elapsed_min": round(s.get("elapsed", 0) / 60, 1),
        "stopped_by": s.get("stopped_by"), "programs": progs,
        "programs_per_dollar": round(progs / spent, 1) if spent else None,
        "best": round(max(e["score"] for e in ev), 6) if ev else None,
        "best_at": {str(x): (round(best_at(ev, x), 6) if x <= spent + 1e-9 else None) for x in CHECK_SPEND},
        "dollars_to": {str(t): (round(v, 4) if (v := first_reach(ev, t)) is not None else None) for t in THRESHOLDS},
        "dollars_api_to": {str(t): (round(v, 4) if (v := first_reach(ev, t, "spent_api")) is not None else None)
                           for t in THRESHOLDS},
        "minutes_to": {str(t): (round(v / 60, 2) if (v := first_reach(ev, t, "te")) is not None else None)
                       for t in THRESHOLDS},
        "K": s.get("K"), "N": s.get("N"), "problem": run["problem"],
    }


def matched(all_runs, problem, base="A"):
    """Seed-paired comparison of each other arm against a Sonnet baseline at equal dollars spent."""
    runs = {}
    for r in all_runs.values():
        if r["problem"] == problem and r["summary"].get("seed") in (1, 2, 3):
            runs[f"{r['arm']}-s{r['summary']['seed']}"] = r
    out = {}
    for other in [o for o in ("B", "C", "L", "Am", "Al", "A") if o != base]:
        rows = []
        for seed in (1, 2, 3):
            a, o = runs.get(f"{base}-s{seed}"), runs.get(f"{other}-s{seed}")
            if not a or not o:
                continue
            m = min(a["summary"]["spent"], o["summary"]["spent"])
            pts = [x for x in CHECK_SPEND if x <= m - 0.01] + [m]
            rows.append({"seed": seed, "common_spend": round(m, 4),
                         "diff": {str(round(x, 4)): round(best_at(o["events"], x) - best_at(a["events"], x), 6)
                                  for x in pts}})
        verdict = {}
        if len(rows) == 3:
            common = min(r["common_spend"] for r in rows)
            for x in [x for x in CHECK_SPEND if x <= common - 0.01] + [common]:
                diffs = [best_at(runs[f"{other}-s{r['seed']}"]["events"], x) - best_at(runs[f"{base}-s{r['seed']}"]["events"], x)
                         for r in rows]
                verdict[str(round(x, 4))] = {"diffs": [round(d, 6) for d in diffs],
                                            "beats_on_all_3": all(d > EPS for d in diffs),
                                            "worst_beats_best": min(best_at(runs[f"{other}-s{r['seed']}"]["events"], x) for r in rows)
                                            > max(best_at(runs[f"{base}-s{r['seed']}"]["events"], x) for r in rows)}
        out[other] = {"per_seed": rows, "verdict": verdict}
    return out


def fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def table(rows):
    lines = ["| run | spent $ | API-equiv $ | min | programs | programs/$ | best | $ to 0.99 | $ to 0.999 | $ to 0.9999 | stopped by |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['spent']:.3f} | {fmt(r['spent_api'], 3)} | {r['elapsed_min']} | {r['programs']} | "
                     f"{fmt(r['programs_per_dollar'], 0)} | {fmt(r['best'], 5)} | {fmt(r['dollars_to']['0.99'])} | "
                     f"{fmt(r['dollars_to']['0.999'])} | {fmt(r['dollars_to']['0.9999'])} | {r['stopped_by']} |")
    return "\n".join(lines)


def arm_table(runs, rows):
    lines = ["| arm | seeds | mean spent $ | programs/$ (mean) | best (mean, range) | $ to 0.99 (mean, range) | $ to 0.999 (mean, range) | $ to 0.9999 (mean, range) |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ("A", "Am", "Al", "B", "C", "L"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue

        def mr(vals, nd=4):
            v = [x for x in vals if x is not None]
            if not v:
                return "—"
            miss = len(vals) - len(v)
            s = f"{np.mean(v):.{nd}f} ({min(v):.{nd}f}–{max(v):.{nd}f})"
            return s + (f", {miss} never" if miss else "")
        lines.append(f"| {ARMS[arm]} | {len(rs)} | {np.mean([r['spent'] for r in rs]):.3f} | "
                     f"{np.mean([r['programs_per_dollar'] or 0 for r in rs]):.0f} | {mr([r['best'] for r in rs], 5)} | "
                     f"{mr([r['dollars_to']['0.99'] for r in rs])} | {mr([r['dollars_to']['0.999'] for r in rs])} | "
                     f"{mr([r['dollars_to']['0.9999'] for r in rs])} |")
    return "\n".join(lines)


def band(ax, group, key, xs, color, label, xscale=1.0):
    ys = np.array([[best_at(r["events"], x, key) if x <= final_x(r, key) + 1e-9 else np.nan for x in xs]
                   for r in group])
    full = ~np.isnan(ys).any(axis=0)       # mean only where every seed is still running
    if not full.any():
        return
    px = xs[full] / xscale
    ax.fill_between(px, np.nanmin(ys[:, full], axis=0), np.nanmax(ys[:, full], axis=0),
                    color=color, alpha=0.18, linewidth=0)
    ax.plot(px, np.nanmean(ys[:, full], axis=0), color=color, linewidth=2, label=label)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot(all_runs, problem):
    runs = {k: v for k, v in all_runs.items() if v["problem"] == problem}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {arm: [r for r in runs.values() if r["arm"] == arm] for arm in ("A", "Am", "Al", "B", "C", "L")}
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="#fcfcfb")
    xs = np.geomspace(0.003, 3.3, 400)
    ts = np.geomspace(0.05, 60, 400)
    for arm, g in groups.items():
        band(axes[0], g, "spent", xs, COLORS[arm], f"{ARMS[arm]} (n={len(g)})")
        band(axes[1], g, "spent", xs, COLORS[arm], ARMS[arm])
        band(axes[2], g, "te", ts * 60, COLORS[arm], ARMS[arm], xscale=60)
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        style(ax)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("dollars spent (list price, as billed)", color="#52514e")
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.05, 60)
    axes[2].set_xlabel("wall-clock minutes", color="#52514e")
    for ax in axes:
        ax.axhline(1.0, color="#9a9994", linewidth=1, linestyle=(0, (4, 3)))
    axes[1].text(0.004, 1.0, "best known" if problem == "circle" else "AlphaEvolve", fontsize=8.5,
                 color="#52514e", va="top")
    axes[0].set_ylabel("best score (sum of radii / 2.635983)" if problem == "circle"
                       else "best score (1.0 = AlphaEvolve)", color="#52514e")
    axes[0].set_title("Best score against spend", loc="left", fontsize=11, color="#0b0b0b")
    axes[1].set_title("Same, zoomed to the top", loc="left", fontsize=11, color="#0b0b0b")
    finals = [max((e["score"] for e in r["events"]), default=0) for g in groups.values() for r in g]
    lo = 0.98 if problem == "circle" else max(0.0, min(finals) - 0.05)
    axes[1].set_ylim(lo, max(1.0005, max(finals) + 0.005))
    fig.suptitle(PROBLEM_NAMES[problem], x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    axes[2].set_title("Best score against wall clock", loc="left", fontsize=11, color="#0b0b0b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, 0.105, "Line: mean over seeds, drawn while every seed is still running. Band: lowest to highest seed.",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / ("curves.png" if problem == "circle" else f"curves_{problem}.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_tuning(runs):
    t = sorted((r for r in runs.values() if r["arm"] == "T"), key=lambda r: r["summary"]["run"])
    if not t:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xs = np.geomspace(0.003, 1.1, 300)
    for i, r in enumerate(t):
        s = r["summary"]
        ys = [best_at(r["events"], x) if x <= s["spent"] + 1e-9 else np.nan for x in xs]
        ax.plot(xs, ys, color=COLORS[f"T{i}"], linewidth=2, label=f"K={s['K']}, N={s['N']}")
    style(ax)
    ax.set_xscale("log")
    ax.set_ylim(0.9, 1.0005)
    ax.set_xlabel("dollars spent", color="#52514e")
    ax.set_ylabel("best score", color="#52514e")
    ax.set_title("Swarm shape, one seed each (exploratory)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "tuning.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


LEGEND = """# Haiku swarm vs Sonnet loop at equal spend: results tables

- **Question.** Does a swarm of cheap Haiku 5.5 calls find better programs than a Sonnet 5.5 loop for the
  same dollars? The pre-set test: B beats the Sonnet loop at matched spend on all three seeds. The Sonnet
  baseline is Al (effort low, 3 seeds per problem); Sonnet's effort level made no visible difference on
  circle packing, so A (effort high) has fewer runs and is shown for reference only.

Generated by `analyze.py` from the runs finished so far; an arm with fewer than 3 seeds is partial, and
a missing arm has no finished run yet. Runs finish in no particular seed order. **Matched spend** compares
two runs' best scores at the same dollars spent; **seed-paired** means run Al-sN against B-sN. The verdict
tables ("X minus Al at matched spend") appear for an arm once all three of its seed pairs have finished;
until all three Al seeds exist for a problem, A (Sonnet at effort high) stands in as the baseline:
"beats on all 3" means a difference above 1e-6 on every seed, "worst beats best" means the arm's worst seed
scores above the baseline's best seed.
Plots, written once a problem has runs: `curves.png` (circle packing), `curves_heilbronn.png`,
`curves_hexagon.png`, `tuning.png`.

- **Problems.** Circle packing: place 26 circles in a unit square, maximize the sum of radii. Heilbronn
  triangle: place 11 points in an equilateral triangle, maximize the smallest area of a triangle any three of
  them form. Hexagon packing: fit 11 unit hexagons in the smallest regular hexagon; scored as 1 / outer side
  length, divided by AlphaEvolve's value, so a smaller outer hexagon scores higher. A program builds the answer; an evaluator scores it.
- **Score.** Circle packing: sum of radii / 2.635983, the best known. Heilbronn and hexagon: the score from the
  evaluator in OpenEvolve's AlphaEvolve examples, where 1.0 = the construction DeepMind's AlphaEvolve
  published. Higher is better; a score above 1.0 beats that reference. **best** is the top score of any program a run evaluated.
- **Arms.** A / Am / Al: stock OpenEvolve (an open-source evolutionary code-search framework) driving Sonnet
  at effort high (Sonnet's API default) / medium / low. Effort sets how much the model thinks per call.
  B: each round runs K=4 swarms in parallel; a swarm is N=8 Haiku proposers (each reads one parent program
  and returns a short idea, no code, effort low) and one Haiku writer (effort medium, Haiku's API default)
  that turns the ideas into one complete program. Each round adds K scored programs to the pool parents
  are drawn from. C: as B, with Sonnet (effort high) as the writer. L: as B, with the writer at effort low.
  T: B with other K and N, circle packing only, on seed 11 so it stays apart from the main seeds 1-3.
  Run ids are arm, then seed (`B-s2`); circle-packing runs have no prefix, `ht-` is Heilbronn, `hx-` hexagon.
- **Stopping.** A run stops when **spent $** reaches $3 (T: $1), "stopped by: cap", or after 60 minutes of run
  time (T: 30), "stopped by: wall". Waits for the subscription's five-hour quota window are not run time.
  Calls in flight at the cap, and a swarm round in flight at the wall, still finish, so spend or time can
  pass the limit slightly. The Haiku swarms usually hit the 60-minute limit at $1-1.5, before the cap, so
  matched-spend comparisons only reach as far as the swarm's final spend.
- **spent $**: list-price cost of each model call's actual token usage. Calls go through `claude -p` (the
  Claude Code CLI in headless mode), the only way to reach the models in this environment; cache writes are
  5-minute (1.25x input). This is the budget axis. **API-equiv $**: the same calls minus the ~1.4K tokens the
  CLI adds to every request: what the raw API would have charged.
- **programs**: distinct programs evaluated; **programs/$** that divided by spent $. **min**: run-time minutes.
  **$ to 0.99** (and 0.999, 0.9999): dollars spent when a run first evaluated a program at or above that
  score; "—" means never. In the per-arm table a "$ to" mean covers only the runs that got there, and
  programs/$ is the mean of the per-run ratios.
"""


def headline(rows, summary):
    """Per problem and arm: the spreads RESULTS.md cites, so each number has one source."""
    out = {}
    for prob in {r["problem"] for r in rows}:
        for arm in {r["arm"] for r in rows if r["problem"] == prob}:
            rs = [r for r in rows if r["problem"] == prob and r["arm"] == arm]
            def span(vals):
                v = [x for x in vals if x is not None]
                return {"min": min(v), "max": max(v), "n": len(v), "never": len(vals) - len(v)} if v else {"n": 0, "never": len(vals)}
            out.setdefault(prob, {})[arm] = {
                "seeds": len(rs),
                "spent": span([r["spent"] for r in rs]),
                "minutes": span([r["elapsed_min"] for r in rs]),
                "best": span([r["best"] for r in rs]),
                "best_at_0.05": span([r["best_at"]["0.05"] for r in rs]),
                "best_at_0.25": span([r["best_at"]["0.25"] for r in rs]),
                "best_at_1": span([r["best_at"]["1.0"] for r in rs]),
                "usd_to_0.99": span([r["dollars_to"]["0.99"] for r in rs]),
                "usd_to_0.999": span([r["dollars_to"]["0.999"] for r in rs]),
                "programs_per_usd": span([r["programs_per_dollar"] for r in rs]),
            }
    out["tuning"] = {r["run"]: {"K": r["K"], "N": r["N"], "usd_to_0.99": r["dollars_to"]["0.99"],
                                "usd_to_0.999": r["dollars_to"]["0.999"], "spent": r["spent"]}
                     for r in rows if r["arm"] == "T"}
    tot = {"all": 0.0, "sonnet": 0.0, "haiku_swarm": 0.0}
    for r in rows:
        tot["all"] += r["spent"]
        tot["sonnet" if r["arm"] in ("A", "Am", "Al", "C") else "haiku_swarm"] += r["spent"]
    out["total_list_usd_48_runs"] = {k: round(v, 2) for k, v in tot.items()}
    api = {"sonnet": 0.0, "haiku_swarm": 0.0}
    for r in rows:
        api["sonnet" if r["arm"] in ("A", "Am", "Al", "C") else "haiku_swarm"] += r["spent_api"]
    out["api_equiv_usd_48_runs"] = {k: round(v, 2) for k, v in api.items()}
    return out


def main():
    runs = load()
    rows = [per_run(r) for r in runs.values()]
    problems_present = [p for p in PROBLEM_NAMES if any(r["problem"] == p for r in rows)]
    def base_for(p):
        # Al is the baseline; until all three Al seeds exist for a problem, A stands in.
        have = {r["run"] for r in rows if r["problem"] == p and r["arm"] == "Al"}
        return "Al" if len(have) == 3 else "A"
    summary = {"runs": rows, "matched": {p: {"baseline": base_for(p), **matched(runs, p, base_for(p))}
                                         for p in problems_present}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "headline.json").write_text(json.dumps(headline(rows, summary), indent=1))
    md = [LEGEND]
    for prob in problems_present:
        prow =[r for r in rows if r["problem"] == prob and r["arm"] in ("A", "Am", "Al", "B", "C", "L")]
        md += [f"# {PROBLEM_NAMES[prob]}", "", "## Per arm", "", arm_table(runs, prow), "", "## Per run", "", table(prow)]
        for other, m in summary["matched"][prob].items():
            if other != "baseline" and m["verdict"]:
                md += ["", f"## {other} minus {summary['matched'][prob]['baseline']} at matched spend (seed-paired best-score differences)", "",
                       "| spend $ | seed 1 | seed 2 | seed 3 | beats on all 3 | worst beats best |",
                       "|---|---|---|---|---|---|"]
                for x, v in m["verdict"].items():
                    md.append(f"| {float(x):.2f} | " + " | ".join(f"{d:+.6f}" for d in v["diffs"]) +
                              f" | {'yes' if v['beats_on_all_3'] else 'no'} | {'yes' if v['worst_beats_best'] else 'no'} |")
        md.append("")
        plot(runs, prob)
    tune_rows = [r for r in rows if r["arm"] == "T"]
    if tune_rows:
        md += ["# Tuning: circle packing, one seed, $1 cap, 30 min", "", table(tune_rows)]
    (OUT / "table.md").write_text("\n".join(md) + "\n")
    plot_tuning(runs)
    print("analysis ->", OUT / "table.md")


if __name__ == "__main__":
    main()
