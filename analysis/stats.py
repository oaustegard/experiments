"""Run statistics and charts for the haiku-cc writeup.

    /opt/anavenv/bin/python analysis/stats.py [--out analysis/out] [--no-fetch]

Reads swarm/state on the hub (ledger, merges, reverts, known) and, from the `fed` remote, the sibling
sessions' own ledgers (hcc/stats/s1..s3: swarm/state/ledger.jsonl and merges.jsonl) and the Haiku-vs-Sonnet
escalation results (hcc/exp/escalation{,2,3}: runs/esc*/results.jsonl). It fetches those branches first
unless --no-fetch. It can be re-run at any point; every number carries the `as_of` time it was taken at and
whether the hub swarm was still running. Writes metrics.json, comparison.md and PNG charts.

Counting rules. A hub ledger record with `ingest` is a candidate commit from a sibling (or from the
escalation run) gated on the hub, not a worker, so it is counted under its source and its cost is not hub
spend. A sibling's local merges (its merges.jsonl) can later be lost when its branch is rebased onto hub
main; the hub's ingest records say which of them landed. `merges on main` is the hub's merges.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import statistics
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "swarm" / "state"
START = dt.datetime(2026, 10, 12, 2, 56, tzinfo=dt.timezone.utc)   # Oskar's "Go"
REMOTE = "fed"
SIBLINGS = ("s1", "s2", "s3")
ESCALATION = (("run 1", "hcc/exp/escalation", "runs/esc"), ("run 2", "hcc/exp/escalation2", "runs/esc2"),
              ("run 3", "hcc/exp/escalation3", "runs/esc3"))
ESC_DESIGN = {"run 1": "40 random failing clusters", "run 2": "4 clusters Haiku had failed",
              "run 3": "10 clusters the swarm abandoned"}

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
    ("13:24", "s1 stops: cross-partition regression"),
    ("18:12", "hub r19: attempt epoch unparks the tail"),
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


# ---------------------------------------------------------------- sources

def git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def fetch() -> list[str]:
    """Fetch the sibling and escalation branches; returns the ones that failed (stale or missing)."""
    failed = []
    for b in [f"hcc/stats/{s}" for s in SIBLINGS] + [b for _, b, _ in ESCALATION]:
        if git("fetch", "-q", REMOTE, f"+{b}:refs/remotes/{REMOTE}/{b}").returncode:
            failed.append(b)
    return failed


def remote_lines(branch: str, path: str) -> list[str] | None:
    """Lines of a file on a fed branch, or None when the branch or file is absent."""
    p = git("show", f"{REMOTE}/{branch}:{path}")
    return p.stdout.splitlines() if p.returncode == 0 else None


def remote_sha(branch: str) -> str | None:
    p = git("rev-parse", "--short=8", f"{REMOTE}/{branch}")
    return p.stdout.strip() if p.returncode == 0 else None


def last_by_id(lines) -> dict:
    last = {}
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        last[d["id"]] = d
    return last


def load_ledger() -> dict:
    return last_by_id((STATE / "ledger.jsonl").open())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def hub_running() -> bool:
    return subprocess.run(["pgrep", "-f", "swarm/run.py run"], capture_output=True).returncode == 0


def source(r: dict) -> str:
    """Which node produced a hub ledger record: 'hub', a sibling name, or 'esc3' and the like."""
    return r.get("sibling") or ("ingest" if r.get("ingest") else "hub")


def kind(r: dict) -> str:
    if r.get("ingest"):
        return "escalation (ingested)" if source(r).startswith("esc") else "sibling (ingested)"
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


def iso(ts: float | None) -> str | None:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(timespec="minutes") if ts else None


def status_counts(recs) -> dict:
    c = collections.Counter(r.get("status") for r in recs)
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def spend(recs) -> float:
    return round(sum(r.get("cost_usd") or 0 for r in recs), 2)


# ---------------------------------------------------------------- metrics

def hub_metrics(led: dict) -> dict:
    own = [r for r in led.values() if not r.get("ingest")]
    merges = read_jsonl(STATE / "merges.jsonl")
    by_src = collections.Counter(source(led.get(m["task"], {})) for m in merges)
    tok = collections.Counter()
    for x in runner_rows():
        for k in ("in", "cache_write", "cache_read", "out_tok", "turns"):
            tok[k] += x.get(k) or 0
    times = [r.get(k) for r in own for k in ("t_launch", "t_done", "t_gate") if r.get(k)]
    times += [m["time"] for m in merges]
    return {
        "workers": len(own), "status": status_counts(own), "spend_usd": spend(own),
        "merges_on_main": len(merges), "merges_on_main_by_source": dict(by_src),
        "reverts": len(read_jsonl(STATE / "reverts.jsonl")),
        "runner_tokens": dict(tok), "last_activity": iso(max(times)) if times else None,
        "_last_ts": max(times) if times else None,
    }


def sibling_metrics(led: dict) -> dict:
    out = {}
    for s in SIBLINGS:
        branch = f"hcc/stats/{s}"
        lines = remote_lines(branch, "swarm/state/ledger.jsonl")
        ingest = [r for r in led.values() if r.get("sibling") == s]
        if lines is None:
            out[s] = {"branch": branch, "missing": True, "ingested_on_hub": status_counts(ingest)}
            continue
        sled = last_by_id(lines)
        merges = [json.loads(l) for l in (remote_lines(branch, "swarm/state/merges.jsonl") or [])]
        lost = sum(1 for m in merges if sled.get(m["task"], {}).get("status") != "merged")
        times = [r.get(k) for r in sled.values() for k in ("t_launch", "t_done", "t_gate") if r.get(k)]
        out[s] = {
            "branch": branch, "sha": remote_sha(branch), "workers": len(sled), "status": status_counts(sled.values()),
            "spend_usd": spend(sled.values()), "turns": sum(r.get("turns") or 0 for r in sled.values()),
            "local_merges": len(merges), "local_merges_lost_on_rebase": lost,
            "ingested_on_hub": status_counts(ingest),
            "landed_on_hub_main": sum(1 for r in ingest if r.get("status") in ("merged", "reverted")),
            "first": iso(min(times)) if times else None, "last": iso(max(times)) if times else None,
        }
    return out


def escalation_metrics() -> dict:
    out = {}
    for name, branch, d in ESCALATION:
        lines = remote_lines(branch, f"{d}/results.jsonl")
        if lines is None:
            out[name] = {"branch": branch, "missing": True}
            continue
        rows = {}
        for l in lines:
            r = json.loads(l)
            rows[(r["key"], r["arm"], r["rep"])] = r          # last record per (cluster, arm, rep) wins
        arms = {}
        for arm in sorted({k[1] for k in rows}, key=lambda a: ("haiku", "haiku-high", "sonnet").index(a)):
            rs = [r for k, r in rows.items() if k[1] == arm]
            solved = sum(bool(r.get("solved")) for r in rs)
            cost = sum(r.get("cost") or 0 for r in rs)
            arms[arm] = {
                "runs": len(rs), "solved": solved, "solve_rate": round(solved / len(rs), 3),
                "clusters_solved": len({r["key"] for r in rs if r.get("solved")}),
                "usd": round(cost, 2), "usd_per_solve": round(cost / solved, 3) if solved else None,
                "median_turns": statistics.median(r.get("turns") or 0 for r in rs),
                "median_secs": statistics.median(r.get("secs") or 0 for r in rs),
            }
        clusters = sorted({k[0] for k in rows})
        solved_by = {c: {a for (k, a, _), r in rows.items() if k == c and r.get("solved")} for c in clusters}
        only_sonnet = sorted(c for c, s in solved_by.items() if "sonnet" in s and not s & {"haiku", "haiku-high"})
        only_haiku = sorted(c for c, s in solved_by.items() if "sonnet" not in s and s & {"haiku", "haiku-high"})
        out[name] = {"branch": branch, "sha": remote_sha(branch), "design": ESC_DESIGN[name],
                     "clusters": len(clusters), "arms": arms, "only_sonnet": only_sonnet,
                     "only_haiku_arms": only_haiku, "neither": sorted(c for c, s in solved_by.items() if not s)}
    return out


def metrics(fetch_failed: list[str]) -> dict:
    led = load_ledger()
    now = dt.datetime.now(dt.timezone.utc)
    running = hub_running()
    hub = hub_metrics(led)
    sib = sibling_metrics(led)
    esc = escalation_metrics()
    known = json.loads((STATE / "known.json").read_text())

    by_kind = collections.defaultdict(collections.Counter)
    for r in led.values():
        by_kind[kind(r)][r.get("status")] += 1
    reasons = collections.Counter()
    for r in led.values():
        if r.get("status") in ("rejected", "failed") and not r.get("ingest"):
            reasons[(r.get("reason") or "")[:40].split(":")[0]] += 1
    suites = collections.defaultdict(lambda: [0, 0])
    for t, rec in known.items():
        s = suites[t.split("/", 1)[0]]
        s[1] += 1
        s[0] += rec.get("status") == "PASS"
    loc = subprocess.run("find hcc -name '*.py' -not -path '*/selftest/*' | xargs cat | wc -l",
                         shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    sib_ok = [v for v in sib.values() if not v.get("missing")]
    end_ts = now.timestamp() if running or not hub["_last_ts"] else hub["_last_ts"]
    hub.pop("_last_ts")
    total = {
        "workers": hub["workers"] + sum(v["workers"] for v in sib_ok),
        "spend_usd": round(hub["spend_usd"] + sum(v["spend_usd"] for v in sib_ok), 2),
        "merges_on_main": hub["merges_on_main"],
        "sibling_local_merges": sum(v["local_merges"] for v in sib_ok),
        "siblings_counted": [s for s, v in sib.items() if not v.get("missing")],
    }
    return {
        "as_of": now.isoformat(timespec="minutes"),
        "hub_running": running,
        "hub_head": git("rev-parse", "--short=8", "HEAD").stdout.strip(),
        "fetch_failed": fetch_failed,
        "elapsed_h": round(hours(end_ts), 2),
        "hub": hub, "siblings": sib, "total": total, "escalation": esc,
        # legacy keys (the 16:53Z writeup draft read these)
        "hub_workers": hub["workers"], "merges": hub["merges_on_main"], "spend_usd": hub["spend_usd"],
        "workers_by_kind": {k: dict(v) for k, v in by_kind.items()},
        "runner": hub["runner_tokens"], "top_loss_reasons": reasons.most_common(8),
        "tests_tracked": len(known), "tests_passing": sum(v[0] for v in suites.values()),
        "suites": {k: {"pass": v[0], "total": v[1]} for k, v in sorted(suites.items())},
        "hcc_lines": int(loc or 0),
        "selftests": len(list((ROOT / "hcc" / "selftest").glob("*.c"))),
    }


# ---------------------------------------------------------------- charts

def charts(out: Path, m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = f"as of {m['as_of'].replace('+00:00', 'Z')}" + (" (hub still running)" if m["hub_running"] else "")
    led = load_ledger()

    # 1. cumulative merges on hub main, split by source, with harness events
    merges = sorted(read_jsonl(STATE / "merges.jsonl"), key=lambda x: x["time"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.step([hours(x["time"]) for x in merges], range(1, len(merges) + 1), where="post", color="#2b6cb0",
            label="all merges on hub main")
    sib = [x for x in merges if source(led.get(x["task"], {})) != "hub"]
    ax.step([hours(x["time"]) for x in sib], range(1, len(sib) + 1), where="post", color="#dd6b20",
            label="of which from siblings / escalation")
    ax.set_ylim(0, len(merges) * 1.05 + 1)
    for hhmm, label in EVENTS:
        h, mi = map(int, hhmm.split(":"))
        x = hours(START.replace(hour=h, minute=mi).timestamp())
        ax.axvline(x, color="#999", lw=0.7, ls=":")
        ax.text(x, ax.get_ylim()[1] * 0.98, label, rotation=90, va="top", ha="right", fontsize=7, color="#555")
    ax.set_xlabel("hours since start (02:56 UTC)")
    ax.set_ylabel("merged fixes (cumulative)")
    ax.set_title(f"Merges on hub main over the run, with harness changes ({stamp})", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "merges_timeline.png", dpi=150)
    plt.close(fig)

    # 2. worker outcomes per node (hub split by worker kind)
    statuses = ["merged", "rejected", "failed", "other"]
    colors = {"merged": "#38a169", "rejected": "#dd6b20", "failed": "#a0aec0", "other": "#e2e8f0"}
    bars = {}
    for k in ("standard", "feature", "rebase"):
        c = m["workers_by_kind"].get(k, {})
        bars[f"hub\n{k}"] = c
    for s, v in m["siblings"].items():
        if not v.get("missing"):
            bars[s] = v["status"]

    def bucket(c: dict, s: str) -> int:
        if s == "other":
            return sum(n for st, n in c.items() if st not in ("merged", "rejected", "failed", "reverted"))
        return c.get(s, 0) + (c.get("reverted", 0) if s == "rejected" else 0)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(bars)
    bottom = [0] * len(names)
    for s in statuses:
        vals = [bucket(bars[n], s) for n in names]
        ax.bar(names, vals, bottom=bottom, label=s if s != "other" else "in flight", color=colors[s])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("workers")
    ax.set_title(f"Worker outcomes by node and worker type ({stamp})", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "outcomes_by_kind.png", dpi=150)
    plt.close(fig)

    # 3. cumulative spend per node
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = {"hub": [r for r in led.values() if not r.get("ingest")]}
    for s in SIBLINGS:
        lines = remote_lines(f"hcc/stats/{s}", "swarm/state/ledger.jsonl")
        if lines is not None:
            series[s] = list(last_by_id(lines).values())
    for name, recs in series.items():
        recs = sorted((r for r in recs if r.get("t_launch")), key=lambda r: r["t_launch"])
        xs, ys, acc = [], [], 0.0
        for r in recs:
            acc += r.get("cost_usd") or 0
            xs.append(hours(r["t_launch"]))
            ys.append(acc)
        ax.plot(xs, ys, label=f"{name} ${acc:.2f}")
    ax.set_xlabel("hours since start")
    ax.set_ylabel("Haiku worker spend, USD (API list price)")
    ax.set_title(f"Worker spend: ${m['total']['spend_usd']:.2f} in all ({stamp})", fontsize=10)
    ax.legend(frameon=False)
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
    ax.set_title(f"hcc test suites ({stamp})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "suites.png", dpi=150)
    plt.close(fig)

    # 5. escalation: solve rate and cost per solve, per run and arm
    esc = {k: v for k, v in m["escalation"].items() if not v.get("missing")}
    if esc:
        arm_colors = {"haiku": "#2b6cb0", "haiku-high": "#63b3ed", "sonnet": "#d69e2e"}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        runs = list(esc)
        width = 0.27
        for j, arm in enumerate(("haiku", "haiku-high", "sonnet")):
            xs = [i + (j - 1) * width for i in range(len(runs))]
            sr = [100 * esc[r]["arms"][arm]["solve_rate"] if arm in esc[r]["arms"] else 0 for r in runs]
            cps = [esc[r]["arms"][arm]["usd_per_solve"] or 0 if arm in esc[r]["arms"] else 0 for r in runs]
            a1.bar(xs, sr, width, label=arm, color=arm_colors[arm])
            a2.bar(xs, cps, width, label=arm, color=arm_colors[arm])
        labels = [f"{r}\n" + textwrap.fill(esc[r]["design"], 18) for r in runs]
        for a in (a1, a2):
            a.set_xticks(range(len(runs)))
            a.set_xticklabels(labels, fontsize=7)
        a1.set_ylim(0, 105)
        a1.set_ylabel("runs solved, %")
        a1.set_title("Solve rate (2 reps per cluster)", fontsize=10)
        a2.set_ylabel("USD per solved run")
        a2.set_title("Cost per solve", fontsize=10)
        a2.legend(frameon=False)
        fig.suptitle("Haiku 5.5 vs Sonnet 5.5 on the same failing clusters, graded in isolation", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "escalation.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- tables

def comparison(m: dict) -> str:
    hub, tot = m["hub"], m["total"]
    sib = {s: v for s, v in m["siblings"].items() if not v.get("missing")}
    sib_workers = sum(v["workers"] for v in sib.values())
    sib_usd = sum(v["spend_usd"] for v in sib.values())
    clock = "so far" if m["hub_running"] else "until the hub swarm stopped"
    ours = {
        "model": "Haiku 5.5 workers (Opus 5.5 for the 3 foundation agents and coordination)",
        "parallel agents": "6-24 per container; hub + 3 sibling containers from 11:27",
        "sessions": f"{tot['workers']:,} Haiku workers ({hub['workers']:,} hub + {sib_workers:,} on {len(sib)} siblings)",
        "cost": (f"${tot['spend_usd']:.2f} Haiku workers (hub ${hub['spend_usd']:.2f}, siblings ${sib_usd:.2f})"
                 " + $14.42 Opus foundation, API list price; Opus coordination not metered"),
        "tokens": (f"hub only: {hub['runner_tokens'].get('cache_read', 0)/1e6:.0f}M cache-read, "
                   f"{hub['runner_tokens'].get('out_tok', 0)/1e6:.1f}M output"),
        "wall clock": f"{m['elapsed_h']:.1f} h {clock}",
        "language": "Python 3.11 (stdlib only), PyPy for large inputs",
        "size": f"{m['hcc_lines']:,} lines + {m['selftests']} selftests",
        "assembler/linker": "system as + gcc as linker driver",
        "headline result": "CPython 3.12, SQLite, mbedtls, Redis, QuickJS, Lua, Doom, tcc; no Linux attempt",
        "test suites": f"{m['tests_passing']:,}/{m['tests_tracked']:,} tracked tests pass",
        "coordination": ("orchestrator queue by failure signature, per-worker worktree, serial no-regression gate; "
                         f"{tot['merges_on_main']} merges on main"),
    }
    head = (f"haiku-cc numbers as of {m['as_of'].replace('+00:00', 'Z')}"
            + (" (hub swarm still running)" if m["hub_running"] else "") + f", hub HEAD {m['hub_head']}.\n\n")
    lines = ["| | Carlini (Opus 4.6) | haiku-cc |", "|---|---|---|"]
    lines += [f"| {k} | {CARLINI[k]} | {ours[k]} |" for k in CARLINI]
    return head + "\n".join(lines) + "\n\n" + nodes_table(m) + "\n" + escalation_table(m)


def nodes_table(m: dict) -> str:
    hub = m["hub"]
    rows = ["| node | workers | merged | rejected | failed | spend | local merges | landed on hub main |",
            "|---|---|---|---|---|---|---|---|"]
    st = hub["status"]
    own = hub["merges_on_main_by_source"].get("hub", 0)
    rows.append(f"| hub | {hub['workers']} | {st.get('merged', 0)} | {st.get('rejected', 0) + st.get('reverted', 0)} "
                f"| {st.get('failed', 0)} | ${hub['spend_usd']:.2f} | {own} | {own} |")
    for s, v in m["siblings"].items():
        if v.get("missing"):
            rows.append(f"| {s} | (ledger not on {v['branch']}) | | | | | | |")
            continue
        st = v["status"]
        rows.append(f"| {s} | {v['workers']} | {st.get('merged', 0)} | {st.get('rejected', 0)} | {st.get('failed', 0)} "
                    f"| ${v['spend_usd']:.2f} | {v['local_merges']} | {v['landed_on_hub_main']} |")
    esc_landed = sum(n for src, n in hub["merges_on_main_by_source"].items() if src.startswith("esc"))
    t = m["total"]
    rows.append(f"| escalation patches | | | | | | | {esc_landed} |")
    rows.append(f"| **total** | **{t['workers']}** | | | | **${t['spend_usd']:.2f}** | "
                f" | **{t['merges_on_main']}** |")
    return "\n".join(rows) + "\n"


def escalation_table(m: dict) -> str:
    rows = ["| run | design | arm | runs | solved | $ total | $ per solve | median turns | median s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, e in m["escalation"].items():
        if e.get("missing"):
            rows.append(f"| {name} | (results not on {e['branch']}) | | | | | | | |")
            continue
        for arm, a in e["arms"].items():
            rows.append(f"| {name} | {e['design']} | {arm} | {a['runs']} | {a['solved']} | {a['usd']:.2f} | "
                        f"{a['usd_per_solve']:.3f} | {a['median_turns']:g} | {a['median_secs']:g} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "analysis" / "out"))
    ap.add_argument("--no-fetch", action="store_true", help="use the fed/* refs already fetched")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = metrics([] if a.no_fetch else fetch())
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    (out / "comparison.md").write_text(comparison(m))
    charts(out, m)
    print(json.dumps({"as_of": m["as_of"], "hub_running": m["hub_running"], "elapsed_h": m["elapsed_h"],
                      "total": m["total"], "hub_spend": m["hub"]["spend_usd"], "tests_passing": m["tests_passing"],
                      "tests_tracked": m["tests_tracked"], "hcc_lines": m["hcc_lines"],
                      "fetch_failed": m["fetch_failed"]}))


if __name__ == "__main__":
    main()
