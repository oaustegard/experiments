"""Run statistics and charts for the haiku-cc writeup.

    /opt/anavenv/bin/python analysis/stats.py [--out analysis/out]

Reads only swarm/state (hub ledger, merges, history, known, runner ledgers) and git, so it can be
re-run at any point of the run. Sibling sessions' own worker ledgers live in their containers; the
hub sees their accepted work as `ingest` records. Writes metrics.json, comparison.md and PNG charts.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "swarm" / "state"
START = dt.datetime(2026, 10, 12, 2, 56, tzinfo=dt.timezone.utc)   # Oskar's "Go"

# Harness changes during the run (UTC), from the session log and commit history.
EVENTS = [
    ("03:25", "Haiku swarm launched (Opus-built foundation)"),
    ("04:33", "queue drained: conformance suites saturated"),
    ("05:34", "fix: re-check merged cluster (stale-state ghosts)"),
    ("07:06", "feature mode for whole-feature clusters"),
    ("07:58", "fix: skipped test is not a regression"),
    ("08:34", "conflicts become rebase tasks"),
    ("09:24", "gate retune; real programs first"),
    ("11:01", "PyPy for large inputs"),
    ("11:27", "3 sibling sessions (federation)"),
    ("12:20", "disk full (worker scratch)"),
]

CARLINI = {  # anthropic.com/engineering/building-c-compiler (Feb 2026), as reported there
    "model": "Opus 4.6", "parallel agents": "16", "sessions": "nearly 2,000",
    "cost": "just under $20,000", "tokens": "2B input, 140M output", "wall clock": "two weeks",
    "language": "Rust (std only)", "size": "100,000 lines",
    "assembler/linker": "GCC's (own ones 'still somewhat buggy')",
    "headline result": "builds Linux 6.9 on x86, ARM, RISC-V; QEMU, FFmpeg, SQLite, postgres, redis, Doom",
    "test suites": "'99% pass rate on most compiler test suites' incl. GCC torture",
    "coordination": "lock files in current_tasks/, agents pull/merge/push, resolve own conflicts",
}


def load_ledger() -> dict:
    last = {}
    for line in (STATE / "ledger.jsonl").open():
        d = json.loads(line)
        last[d["id"]] = d
    return last


def kind(r: dict) -> str:
    if r.get("ingest"):
        return "sibling (ingested)"
    if r.get("rebase_of"):
        return "rebase"
    if r.get("feature"):
        return "feature"
    return "standard"


def runner_rows() -> list[dict]:
    rows = []
    for f in glob.glob(str(STATE / "tasks" / "*" / "fanout" / "ledger.jsonl")):
        last = None
        for line in open(f):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
        if last:
            rows.append(last)
    return rows


def hours(ts: float) -> float:
    return (ts - START.timestamp()) / 3600


def metrics() -> dict:
    led = load_ledger()
    rows = runner_rows()
    merges = [json.loads(l) for l in (STATE / "merges.jsonl").open()]
    known = json.loads((STATE / "known.json").read_text())
    by_kind = collections.defaultdict(collections.Counter)
    for r in led.values():
        by_kind[kind(r)][r.get("status")] += 1
    reasons = collections.Counter()
    for r in led.values():
        if r.get("status") in ("rejected", "failed"):
            reasons[(r.get("reason") or "")[:40].split(":")[0]] += 1
    suites = collections.defaultdict(lambda: [0, 0])
    for t, rec in known.items():
        s = suites[t.split("/", 1)[0]]
        s[1] += 1
        s[0] += rec.get("status") == "PASS"
    tok = collections.Counter()
    for x in rows:
        for k in ("in", "cache_write", "cache_read", "out_tok", "turns"):
            tok[k] += x.get(k) or 0
    loc = subprocess.run("find hcc -name '*.py' -not -path '*/selftest/*' | xargs cat | wc -l",
                         shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "elapsed_h": round(hours(dt.datetime.now(dt.timezone.utc).timestamp()), 2),
        "hub_workers": len(led), "workers_by_kind": {k: dict(v) for k, v in by_kind.items()},
        "merges": len(merges), "spend_usd": round(sum(r.get("cost_usd") or 0 for r in led.values()), 2),
        "runner": dict(tok), "top_loss_reasons": reasons.most_common(8),
        "tests_tracked": len(known), "tests_passing": sum(v[0] for v in suites.values()),
        "suites": {k: {"pass": v[0], "total": v[1]} for k, v in sorted(suites.items())},
        "hcc_lines": int(loc or 0),
        "selftests": len(list((ROOT / "hcc" / "selftest").glob("*.c"))),
    }


def charts(out: Path, m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    led = load_ledger()
    merges = sorted(json.loads(l)["time"] for l in (STATE / "merges.jsonl").open())
    # 1. cumulative merges with harness events
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step([hours(t) for t in merges], range(1, len(merges) + 1), where="post", color="#2b6cb0")
    for hhmm, label in EVENTS:
        h, mi = map(int, hhmm.split(":"))
        x = hours(START.replace(hour=h, minute=mi).timestamp())
        ax.axvline(x, color="#999", lw=0.7, ls=":")
        ax.text(x, ax.get_ylim()[1] * 0.98, label, rotation=90, va="top", ha="right", fontsize=7, color="#555")
    ax.set_xlabel("hours since start (02:56 UTC)")
    ax.set_ylabel("merged fixes (cumulative)")
    ax.set_title("Haiku swarm merges over the run, with harness changes")
    fig.tight_layout()
    fig.savefig(out / "merges_timeline.png", dpi=150)
    plt.close(fig)

    # 2. worker outcomes by worker kind
    kinds = ["standard", "feature", "rebase", "sibling (ingested)"]
    statuses = ["merged", "rejected", "failed"]
    colors = {"merged": "#38a169", "rejected": "#dd6b20", "failed": "#a0aec0"}
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0] * len(kinds)
    for s in statuses:
        vals = [m["workers_by_kind"].get(k, {}).get(s, 0) for k in kinds]
        ax.bar(kinds, vals, bottom=bottom, label=s, color=colors[s])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("workers")
    ax.set_title("Worker outcomes by worker type")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "outcomes_by_kind.png", dpi=150)
    plt.close(fig)

    # 3. cumulative spend vs merges
    recs = sorted((r for r in led.values() if r.get("t_launch")), key=lambda r: r["t_launch"])
    xs, spend, acc = [], [], 0.0
    for r in recs:
        acc += r.get("cost_usd") or 0
        xs.append(hours(r["t_launch"]))
        spend.append(acc)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, spend, color="#805ad5")
    ax.set_xlabel("hours since start")
    ax.set_ylabel("Haiku worker spend, USD (API list price)")
    ax.set_title(f"Cumulative worker spend: ${m['spend_usd']:.2f} for {m['merges']} merges")
    fig.tight_layout()
    fig.savefig(out / "spend.png", dpi=150)
    plt.close(fig)

    # 4. per-suite pass rate now
    names = sorted(m["suites"], key=lambda s: -m["suites"][s]["total"])
    rate = [100 * m["suites"][s]["pass"] / m["suites"][s]["total"] for s in names]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], rate[::-1], color="#2b6cb0")
    for i, s in enumerate(names[::-1]):
        ax.text(101, i, f"{m['suites'][s]['pass']:,}/{m['suites'][s]['total']:,}", va="center", fontsize=7)
    ax.set_xlim(0, 125)
    ax.set_xlabel("% passing (behaviour identical to the gcc-built reference)")
    ax.set_title("hcc test suites")
    fig.tight_layout()
    fig.savefig(out / "suites.png", dpi=150)
    plt.close(fig)


def comparison(m: dict) -> str:
    ours = {
        "model": "Haiku 5.5 workers (Opus 5.5 for the 3 foundation agents and coordination)",
        "parallel agents": "6-24 per container; hub + 3 sibling containers from 11:27",
        "sessions": f"{m['hub_workers']} hub workers (+ sibling workers not counted here)",
        "cost": f"${m['spend_usd']:.2f} Haiku workers + $14.42 Opus foundation (API list price)",
        "tokens": f"{m['runner'].get('cache_read', 0)/1e6:.0f}M cache-read, {m['runner'].get('out_tok', 0)/1e6:.1f}M output",
        "wall clock": f"{m['elapsed_h']:.1f} h so far",
        "language": "Python 3.11 (stdlib only), PyPy for large inputs",
        "size": f"{m['hcc_lines']:,} lines + {m['selftests']} selftests",
        "assembler/linker": "system as + gcc as linker driver",
        "headline result": "CPython 3.12, SQLite, mbedtls, Redis, QuickJS, Lua, Doom (88/93), tcc; no Linux attempt",
        "test suites": f"{m['tests_passing']:,}/{m['tests_tracked']:,} tracked tests pass",
        "coordination": "orchestrator queue by failure signature, per-worker worktree, serial no-regression gate",
    }
    lines = ["| | Carlini (Opus 4.6) | haiku-cc |", "|---|---|---|"]
    lines += [f"| {k} | {CARLINI[k]} | {ours[k]} |" for k in CARLINI]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "analysis" / "out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = metrics()
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    (out / "comparison.md").write_text(comparison(m))
    charts(out, m)
    print(json.dumps({k: m[k] for k in ("elapsed_h", "hub_workers", "merges", "spend_usd", "tests_passing",
                                        "tests_tracked", "hcc_lines")}))


if __name__ == "__main__":
    main()
