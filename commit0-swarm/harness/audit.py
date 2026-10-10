"""Flag every agent action that could have reached a library's original source.

    python harness/audit.py RUN [RUN ...]     -> data/runs/RUN/audit.jsonl, summary on stdout

Agents run as root in the main container, so permissions cannot hide anything
from them; the control is this audit. The original source of a library was
reachable, at some point in the run, through:
  - git history (the stub commit's parent is the reference; trees made before
    the history-free checkout carried it),
  - the uv/pip caches (every library was installed from PyPI for its deps),
  - other libraries' venvs (cookiecutter's env holds jinja2 and marshmallow, etc.),
  - the certification trees (WORK/cert/LIB/ref-src, deleted after certifying),
  - other runs' trees and saved patches.
Every path an agent names outside its own tree, the jail's bin, and its own
prompt and out file is reported; git commands that read other revisions are
reported. Transcripts come from RUN/transcripts (cost.py copies them).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BIN, DATA, ENVS, LITE, MIRRORS, WORK  # noqa: E402

PATH = re.compile(r"(/(?:tmp|root|home|opt|usr|var|etc|mnt)/[^\s'\"`;|&)<>]*)")
GIT_HISTORY = re.compile(r"\bgit\b[^|;&]*\b(show|log\s+-p|log\s+--patch|diff\s+\S*(HEAD[~^]|[0-9a-f]{7,})|"
                         r"checkout\s+\S*(HEAD[~^]|[0-9a-f]{7,})|cat-file|rev-list|reflog|archive|blame)\b")
SAFE_PREFIX = ("/tmp/claude-subagent-allowlist.json", "/dev/null")


def severity(path: str, own: Path, run_dir: Path, unit: str) -> str | None:
    """None: fine. 'high': a place that held some library's original or another
    run's work. 'low': anything else outside the agent's tree."""
    p = path.rstrip("/.,:\\")
    lib = unit.split(".")[0]
    if (p.startswith(str(own)) or p.startswith(str(BIN)) or p.startswith(SAFE_PREFIX)
            or p.startswith(str(run_dir / "out")) or p == str(run_dir / "prompts" / f"{unit}.md")
            or p == str(run_dir)
            or p.startswith(str(ENVS / lib)) or p in ("/tmp", "/usr/bin/env")
            or p.startswith(("/tmp/tmp", "/tmp/pytest"))):
        return None
    if p.startswith(str(WORK / run_dir.name)) and not any(
            p.startswith(str(WORK / run_dir.name / other)) for other in LITE):
        return None                                   # scratch files beside the trees
    if (p.startswith((str(ENVS), str(MIRRORS), str(WORK), "/root/.cache", str(DATA / "runs")))
            or "site-packages" in p or ".cache" in p):
        return "high"
    return "low"


def tool_inputs(transcript: Path):
    for line in open(transcript, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message")
        if not isinstance(m, dict) or m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                yield b.get("name"), b.get("input") or {}


def audit_unit(run: str, unit: str, transcript: Path) -> dict:
    run_dir = DATA / "runs" / run
    own = WORK / run / unit.split(".")[0]
    outside, high, history = [], [], []
    for name, inp in tool_inputs(transcript):
        text = json.dumps(inp)
        for p in PATH.findall(text.replace("\\n", " ")):
            sev = severity(p, own, run_dir, unit)
            if sev:
                (high if sev == "high" else outside).append(f"{name}: {p}")
        if name == "Bash" and GIT_HISTORY.search(inp.get("command", "")):
            history.append(inp["command"][:200])
    return {"unit": unit, "high": sorted(set(high)), "low": sorted(set(outside)), "git_history": history}


def main(runs: list[str]):
    for run in runs:
        d = DATA / "runs" / run
        rows = [audit_unit(run, t.stem, t) for t in sorted((d / "transcripts").glob("*.jsonl"))]
        (d / "audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        flagged = [r for r in rows if r["high"] or r["git_history"]]
        print(f"{run}: {len(rows)} transcripts, {len(flagged)} high-severity, "
              f"{sum(bool(r['low']) for r in rows)} with other outside paths")
        for r in flagged:
            print(f"  {r['unit']}: high={r['high'][:8]} history={r['git_history'][:3]}")


if __name__ == "__main__":
    main(sys.argv[1:])
