"""Extract real delegations from archived Claude Code sessions, with the context each one had.

A delegation is a tool call that hands a task to another agent: Agent (not the
`fork` type, which inherits the whole context), create_session and send_message.
Its context is every chunk of the session before the call, chunked and
secret-scrubbed by subagent-context-filter/chunking.py, the same code the
shipped filter uses.

Usage: EVAL_DATA=<private dir> python3 prep.py
Reads  <EVAL_DATA>/raw/<session8>.jsonl
Writes <EVAL_DATA>/chunks/<session8>.json   (chunks; transcript text, private)
       <EVAL_DATA>/delegations.jsonl        (one row per delegation; task text, private)
       results/prep.json                    (counts only, safe to commit)
"""
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("subagent-context-filter")))
import chunking  # noqa: E402
import jevfilter  # noqa: E402

DATA = Path(os.environ.get("EVAL_DATA") or HERE / "data")
DELEG_TOOLS = ("Agent", "Task")
DELEG_SUFFIXES = ("create_session", "send_message")


def is_delegation(chunk):
    if chunk["kind"] != "tool":
        return False
    name = chunk.get("tool", "")
    return name in DELEG_TOOLS or name.endswith(DELEG_SUFFIXES)


def task_of(chunk):
    try:
        inp = json.loads(chunk.get("input") or "{}")
    except ValueError:
        return None, None
    if inp.get("subagent_type") == "fork":
        return None, "fork"
    text = inp.get("prompt") or inp.get("message") or ""
    return (text.strip() or None), inp.get("subagent_type") or chunk.get("tool")


def main():
    (DATA / "chunks").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], {"fork": 0, "no_task": 0, "no_context": 0}
    sessions_used = 0
    for f in sorted((DATA / "raw").glob("*.jsonl")):
        chunks, _ = chunking.chunk_transcript(str(f))
        found = False
        for c in chunks:
            if not is_delegation(c):
                continue
            task, kind = task_of(c)
            if kind == "fork":
                skipped["fork"] += 1
                continue
            if not task:
                skipped["no_task"] += 1
                continue
            ctx = [x for x in chunks if x["id"] < c["id"] and not is_delegation(x)]
            if len(ctx) < 3:
                skipped["no_context"] += 1
                continue
            ctx_tokens = sum(jevfilter.tokens(jevfilter.render(x)) for x in ctx)
            rows.append({"session": f.stem, "deleg_chunk": c["id"], "agent": kind,
                         "task": task, "n_ctx": len(ctx), "ctx_tokens": ctx_tokens})
            found = True
        if found:
            sessions_used += 1
            (DATA / "chunks" / f"{f.stem}.json").write_text(json.dumps(chunks))
    with open(DATA / "delegations.jsonl", "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    toks = sorted(r["ctx_tokens"] for r in rows)
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))] if toks else None
    summary = {
        "sessions_scanned": len(list((DATA / "raw").glob("*.jsonl"))),
        "sessions_with_delegations": sessions_used,
        "delegations": len(rows), "skipped": skipped,
        "agents": dict(sorted({r["agent"]: sum(1 for x in rows if x["agent"] == r["agent"])
                               for r in rows}.items(), key=lambda kv: -kv[1])),
        "ctx_chunks_median": statistics.median(r["n_ctx"] for r in rows) if rows else None,
        "ctx_tokens_p10_p50_p90_max": [q(0.1), q(0.5), q(0.9), toks[-1] if toks else None],
        "delegations_ctx_over": {str(b): sum(1 for t in toks if t > b)
                                 for b in (5_000, 10_000, 20_000)},
        "task_chars_median": statistics.median(len(r["task"]) for r in rows) if rows else None,
    }
    (HERE / "results" / "prep.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
