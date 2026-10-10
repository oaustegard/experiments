"""Per-spawn cost and tool audit from Agent-tool transcripts.

    python harness/cost.py RUN [--tasks-dir DIR]   -> data/runs/RUN/costs.jsonl

Each transcript is
<tasks-dir>/<agent_id>.output and is copied to RUN/transcripts/ (gitignored).
Pricing and method as temporal-routing-headroom/harness/subagent_cost.py: one
usage per message id; output tokens in these transcripts are streaming-start
values, so out_floor is a floor and the dollar figure is input-side. Haiku 5.5
turns whose prompt exceeds 100K tokens are priced at the long rate card, 5x every rate.
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
LONG = 100_000
LONG_MULT = 5


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
            i, w, r = (u.get("input_tokens", 0), u.get("cache_creation_input_tokens", 0),
                       u.get("cache_read_input_tokens", 0))
            tot["inp"] += i
            tot["cw"] += w
            tot["cr"] += r
            # Haiku 5.5 has two rate cards: a prompt over 100K tokens pays $0.50/$2.50 per MTok
            # against $0.10/$0.50, 5x on every token type (claude-api skill, models.md and
            # model-migration.md, cached 2026-10-06). Until 2026-10-10 this read 2x (from
            # down-skilling 1.7), which under-priced every long turn.
            mult = LONG_MULT if model == "claude-haiku-5-5" and i + w + r > LONG else 1
            p = PRICE.get(model, (0, 0, 0, 0))
            tot["usd_e6"] += mult * (i * p[0] + w * p[1] + r * p[2])
            tot["long_turns"] += mult > 1
            tot["out_floor"] += u.get("output_tokens", 0)
            tot["turns"] += 1
        elif m.get("role") == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    text = json.dumps(b.get("content"))
                    for g in ("jail_route", "subagent_scope"):
                        if f"{g}:" in text:
                            refusals[g] += 1
    usd = tot.pop("usd_e6") / 1e6
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
    # Each worker's first message names its prompt file, data/runs/RUN/prompts/<id>.md,
    # so transcripts map to tasks without recording agent ids at dispatch.
    agents = {}
    if (d / "agents.jsonl").exists():
        agents = {m["instance_id"]: m["agent_id"] for m in map(json.loads, (d / "agents.jsonl").open())}
    # The live tasks dir only holds this session's agents; a re-price of an older run
    # finds its agents in the previous costs.jsonl and its transcripts in RUN/transcripts.
    if (d / "costs.jsonl").exists():
        for r in map(json.loads, (d / "costs.jsonl").read_text().splitlines()):
            agents.setdefault(r["instance_id"], r["agent_id"])
    marker = f"/runs/{a.run}/prompts/"
    for t in src.glob("*.output"):
        if not t.exists():   # dangling symlink to a finished background command
            continue
        with open(t, errors="replace") as f:
            head = f.read(4000)
        i = head.find(marker)
        if i >= 0:
            iid = head[i + len(marker):].split(".md", 1)[0]
            agents.setdefault(iid, t.stem)
    for m in ({"instance_id": k, "agent_id": v} for k, v in sorted(agents.items())):
        t = src / f"{m['agent_id']}.output"
        kept = d / "transcripts" / f"{m['instance_id']}.jsonl"
        if t.exists() and (not kept.exists() or t.stat().st_size >= kept.stat().st_size):
            shutil.copyfile(t, kept)
        if not kept.exists():
            print("missing transcript", m, file=sys.stderr)
            continue
        rows.append({"instance_id": m["instance_id"], "agent_id": m["agent_id"], **tally(kept)})
    if len(rows) < len(agents):
        sys.exit(f"{len(agents) - len(rows)} transcripts missing; costs.jsonl left as it was")
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
