#!/usr/bin/env python3
"""Build the codex corpus: session digests from the hub's transcripts branch.

Each archived transcript (claude-workspace `transcripts` branch, one commit per
stop) is reduced to what was said: the user's messages and the assistant's
prose, with tool calls kept as one-line names and tool output dropped. Tool
output is most of the bytes and little of the meaning; the memories carry the
measured results.

    python3 corpus.py HUB OUT      # HUB = claude-workspace checkout, OUT = data dir (private)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

NOISE = re.compile(r"<(system-reminder|command-[a-z]+|local-command-stdout|task-notification)>.*?</\1>", re.S)


def latest_blobs(hub: Path) -> dict[str, str]:
    """path -> commit of its newest version on origin/transcripts."""
    log = subprocess.run(["git", "-C", str(hub), "log", "origin/transcripts", "--format=C %H", "--name-only"],
                         capture_output=True, text=True, check=True).stdout
    out, commit = {}, None
    for line in log.splitlines():
        if line.startswith("C "):
            commit = line[2:]
        elif line.strip() and line not in out:
            out[line] = commit
    return out


def _text(content) -> tuple[list[str], list[str]]:
    texts, tools = [], []
    if isinstance(content, str):
        return [content], tools
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            texts.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            name = b.get("name", "?")
            inp = b.get("input") or {}
            hint = inp.get("description") or inp.get("query") or inp.get("file_path") or ""
            tools.append(f"{name}({str(hint)[:80]})" if hint else name)
    return texts, tools


def digest(lines: list[str]) -> tuple[str, dict]:
    parts, meta = [], {"title": None, "turns": 0, "tools": 0, "start": None, "end": None}
    pending_tools: list[str] = []
    for raw in lines:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        t = d.get("type")
        if t == "custom-title":
            meta["title"] = d.get("customTitle")
        if t not in ("user", "assistant") or d.get("isSidechain"):
            continue
        ts = d.get("timestamp")
        if ts:
            meta["start"] = meta["start"] or ts
            meta["end"] = ts
        texts, tools = _text((d.get("message") or {}).get("content"))
        if t == "assistant":
            pending_tools += tools
            meta["tools"] += len(tools)
            body = "\n".join(x for x in texts if x.strip())
            if body:
                if pending_tools:
                    parts.append("  [tools: " + ", ".join(pending_tools[:12]) +
                                 (f" +{len(pending_tools) - 12}" if len(pending_tools) > 12 else "") + "]")
                    pending_tools = []
                parts.append(f"MUNINN [{(ts or '')[:16]}]: {body.strip()}")
        else:
            if d.get("isMeta") or d.get("toolUseResult") is not None:
                continue
            body = NOISE.sub("", "\n".join(texts)).strip()
            if body and not body.startswith("[Request interrupted"):
                meta["turns"] += 1
                if pending_tools:
                    parts.append("  [tools: " + ", ".join(pending_tools[:12]) + "]")
                    pending_tools = []
                parts.append(f"OSKAR [{(ts or '')[:16]}]: {body[:6000]}")
    return "\n\n".join(parts), meta


def main(hub: Path, out: Path) -> int:
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    best: dict[str, dict] = {}
    for path, commit in sorted(latest_blobs(hub).items()):
        blob = subprocess.run(["git", "-C", str(hub), "show", f"{commit}:{path}"], capture_output=True,
                              text=True, errors="replace").stdout
        text, meta = digest(blob.splitlines())
        if meta["turns"] == 0:
            continue
        # A session is archived once per stop under a path that can change with its title;
        # the longest transcript is the latest and contains the others.
        sid = Path(path).stem.split("-")[1][:8]
        if sid in best and best[sid]["raw"] >= len(blob):
            continue
        (out / "sessions" / f"{sid}.txt").write_text(text)
        best[sid] = {"sid": sid, "path": path, "chars": len(text), "raw": len(blob), **meta}
    index = sorted(best.values(), key=lambda r: r["start"] or "")
    with (out / "sessions.jsonl").open("w") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    print(f"{len(index)} sessions, {sum(r['chars'] for r in index) / 1e6:.1f}M digest chars "
          f"from {sum(r['raw'] for r in index) / 1e6:.0f}M raw")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
