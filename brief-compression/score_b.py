#!/usr/bin/env python3
"""Score Task B runs. Usage: python3 score_b.py <runs_root> [replies.json]"""
import hashlib, json, re, sys
from pathlib import Path

CHECKS = ["report_exists", "report_header", "report_five_rows", "report_rows_ok",
          "summary_exists", "summary_keys", "rows_read_ok", "rows_kept_ok", "top_user_ok",
          "log_one_line", "input_unmodified", "no_extra_files", "reply_ok"]

def score_run(r, reply):
    d = Path(r["dir"]); g = r["gold"]; c = {k: False for k in CHECKS}
    rep = d / "report.tsv"
    c["report_exists"] = rep.is_file()
    if c["report_exists"]:
        lines = rep.read_text().split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        c["report_header"] = bool(lines) and lines[0] == "user\tbytes"
        body = lines[1:]
        c["report_five_rows"] = len(body) == 5
        want = [f"{u}\t{b}" for u, b in g["top"]]
        c["report_rows_ok"] = body == want
    s = d / "summary.json"
    c["summary_exists"] = s.is_file()
    if c["summary_exists"]:
        try:
            j = json.loads(s.read_text())
            c["summary_keys"] = isinstance(j, dict) and set(j) == {"rows_read", "rows_kept", "top_user"}
            c["rows_read_ok"] = j.get("rows_read") == g["rows_read"]
            c["rows_kept_ok"] = j.get("rows_kept") == g["rows_kept"]
            c["top_user_ok"] = j.get("top_user") == g["top_user"]
        except Exception:
            pass
    log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
    c["log_one_line"] = log == f"boot ok\nschema ok\nready\n{d.name} ok {g['rows_kept']}\n"
    inp = d / "events.csv"
    c["input_unmodified"] = inp.exists() and hashlib.sha256(inp.read_bytes()).hexdigest() == g["sha256"]
    present = sorted(p.name for p in d.iterdir())
    allowed = {"events.csv", "log.txt", "report.tsv", "summary.json"}
    c["no_extra_files"] = set(present) <= allowed
    c["reply_ok"] = reply is not None and reply.strip().strip("*`.") == f"OK {g['rows_kept']}"
    c["success"] = all(c[k] for k in CHECKS)
    c["n_pass"] = sum(c[k] for k in CHECKS)
    return c

def main():
    root = Path(sys.argv[1])
    briefs = json.loads((root / "briefs.json").read_text())
    replies = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
    scores = {}
    for run, r in briefs.items():
        if run in replies:
            scores[run] = {"style": r["style"], "model": r["model"], **score_run(r, replies[run])}
    (root / "scores.json").write_text(json.dumps(scores, indent=1))
    cells = {}
    for s in scores.values():
        cells.setdefault((s["model"], s["style"]), []).append(s)
    print(f"{'model':7} {'style':12} {'n':>3} {'success':>8} {'mean_pass':>9}  weakest checks")
    for (m, st), v in sorted(cells.items()):
        succ = sum(x["success"] for x in v); mp = sum(x["n_pass"] for x in v) / len(v)
        fails = {k: sum(not x[k] for x in v) for k in CHECKS}
        weak = ", ".join(f"{k}={n}" for k, n in sorted(fails.items(), key=lambda kv: -kv[1]) if n)
        print(f"{m:7} {st:12} {len(v):3d} {succ:5d}/{len(v):<2d} {mp:9.2f}  {weak or '-'}")

if __name__ == "__main__":
    main()
