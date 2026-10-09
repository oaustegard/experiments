"""Per-spawn cost and tool audit from Agent-tool transcripts.

    python harness/cost.py RUN [--tasks-dir DIR]   -> data/runs/RUN/costs.jsonl

RUN/agents.jsonl maps instance_id -> agent_id; each transcript is
<tasks-dir>/<agent_id>.output and is copied to RUN/transcripts/ (gitignored).
Pricing and method as temporal-routing-headroom/harness/subagent_cost.py: one
usage per message id; output tokens in these transcripts are streaming-start
values, so out_floor is a floor and the dollar figure is input-side.
The audit counts tool calls by name and the guard refusals (jail_route,
subagent_scope) each run hit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

PRICE = {  # $/MTok: input, 5m cache write, cache read, output (claude-api skill, 2026-10-06)
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
    "claude-sonnet-5-5": (2.00, 2.50, 0.20, 10.0),
    "claude-opus-5-5": (4.00, 5.00, 0.20, 20.0),
}


def default_tasks_dir() -> Path:
    base = Path("/tmp/claude-0")
    hits = sorted(base.glob("*/*/tasks"), key=lambda p: p.stat().st_mtime) if base.exists() else []
    return hits[-1] if hits else Path(".")


def tally(path: Path) -> dict:
    seen, tot, model = set(), Counter(), None
    tools, refusals = Counter(), Counter()
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        content = m.get("content") if isinstance(m.get("content"), list) else []
        if m.get("role") == "assistant":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools[b.get("name")] += 1
            if m.get("id") in seen:
                continue
            seen.add(m.get("id"))
            u = m.get("usage") or {}
            model = m.get("model") or model
            tot["inp"] += u.get("input_tokens", 0)
            tot["cw"] += u.get("cache_creation_input_tokens", 0)
            tot["cr"] += u.get("cache_read_input_tokens", 0)
            tot["out_floor"] += u.get("output_tokens", 0)
            tot["turns"] += 1
        elif m.get("role") == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    text = json.dumps(b.get("content"))
                    for g in ("jail_route", "subagent_scope"):
                        if f"{g}:" in text:
                            refusals[g] += 1
    p = PRICE.get(model, (0, 0, 0, 0))
    usd = (tot["inp"] * p[0] + tot["cw"] * p[1] + tot["cr"] * p[2]) / 1e6
    return {"model": model, **tot, "usd_input_side": round(usd, 5),
            "tools": dict(tools), "refusals": dict(refusals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tasks-dir", type=Path, default=None)
    a = ap.parse_args()
    d = DATA / "runs" / a.run
    src = a.tasks_dir or default_tasks_dir()
    (d / "transcripts").mkdir(exist_ok=True)
    rows = []
    for m in map(json.loads, (d / "agents.jsonl").open()):
        t = src / f"{m['agent_id']}.output"
        kept = d / "transcripts" / f"{m['instance_id']}.jsonl"
        if t.exists() and (not kept.exists() or t.stat().st_size >= kept.stat().st_size):
            shutil.copyfile(t, kept)
        if not kept.exists():
            print("missing transcript", m, file=sys.stderr)
            continue
        rows.append({"instance_id": m["instance_id"], "agent_id": m["agent_id"], **tally(kept)})
    (d / "costs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    usd = sum(r["usd_input_side"] for r in rows)
    other = Counter()
    for r in rows:
        for k, v in r["tools"].items():
            if k not in ("Bash", "Read", "Edit", "Write", "Grep", "Glob", "SubagentHandback"):
                other[k] += v
    print(f"{len(rows)} spawns, input-side ${usd:.4f} (mean ${usd / max(1, len(rows)):.4f}); "
          f"mean turns {sum(r['turns'] for r in rows) / max(1, len(rows)):.1f}; "
          f"refusals {sum((Counter(r['refusals']) for r in rows), Counter())}; off-list tools {dict(other)}")


if __name__ == "__main__":
    main()
