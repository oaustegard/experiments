#!/usr/bin/env python3
"""Score every run directory against briefs.json.

Usage: python3 score.py <runs_root> [replies.json]

replies.json (optional) maps run id -> the subagent's final message text.
Emits <runs_root>/scores.json and prints a per-cell table.
Nine binary checks per run; success = all nine.
"""
import hashlib, json, sys
from pathlib import Path

CHECKS = ["out_exists", "keys_exact", "count_ok", "sha_ok", "run_ok",
          "log_one_line", "input_unmodified", "no_extra_files", "reply_done"]

def score_run(r, reply):
    d = Path(r["dir"])
    c = {k: False for k in CHECKS}
    out = d / "out.json"
    c["out_exists"] = out.is_file()
    if c["out_exists"]:
        try:
            j = json.loads(out.read_text())
            c["keys_exact"] = isinstance(j, dict) and set(j) == {"count", "sha256", "run"}
            c["count_ok"] = isinstance(j.get("count"), int) and not isinstance(j.get("count"), bool) \
                and j["count"] == r["gold"]["count"]
            c["sha_ok"] = str(j.get("sha256", "")).lower() == r["gold"]["sha256"]
            c["run_ok"] = j.get("run") == r["dir"].rsplit("/", 1)[-1]
        except Exception:
            pass
    log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
    c["log_one_line"] = log == f"boot ok\nschema ok\nready\n{d.name} done\n"
    inp = d / "input.log"
    c["input_unmodified"] = inp.exists() and hashlib.sha256(inp.read_bytes()).hexdigest() == r["gold"]["sha256"]
    c["no_extra_files"] = sorted(p.name for p in d.iterdir()) == ["input.log", "log.txt", "out.json"] \
        if c["out_exists"] else sorted(p.name for p in d.iterdir()) == ["input.log", "log.txt"]
    c["reply_done"] = reply is not None and reply.strip().strip("*`.") == "DONE"
    c["success"] = all(c[k] for k in CHECKS)
    c["n_pass"] = sum(c[k] for k in CHECKS)
    return c

def main():
    root = Path(sys.argv[1])
    briefs = json.loads((root / "briefs.json").read_text())
    replies = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
    scores = {}
    for run, r in briefs.items():
        if run not in replies:
            continue  # not dispatched yet
        scores[run] = {"style": r["style"], "model": r["model"], **score_run(r, replies[run])}
    (root / "scores.json").write_text(json.dumps(scores, indent=1))
    cells = {}
    for s in scores.values():
        cells.setdefault((s["model"], s["style"]), []).append(s)
    print(f"{'model':7} {'style':12} {'n':>3} {'success':>8} {'mean_pass':>9}  weakest checks")
    for (m, st), v in sorted(cells.items()):
        succ = sum(x["success"] for x in v)
        mp = sum(x["n_pass"] for x in v) / len(v)
        fails = {k: sum(not x[k] for x in v) for k in CHECKS}
        weak = ", ".join(f"{k}={n}" for k, n in sorted(fails.items(), key=lambda kv: -kv[1]) if n)
        print(f"{m:7} {st:12} {len(v):3d} {succ:5d}/{len(v):<2d} {mp:9.2f}  {weak or '-'}")

if __name__ == "__main__":
    main()
