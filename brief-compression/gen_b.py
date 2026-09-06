#!/usr/bin/env python3
"""Task B — a denser brief with more places for telegraphese to lose information.

Usage: python3 gen_b.py <runs_root> [--n 10] [--seed 11]

Per run directory: events.csv (300 rows), log.txt (3 lines). The subagent
must produce report.tsv and summary.json, append one log line, and reply
"OK <rows_kept>". Thirteen binary checks (see score_b.py).

Rules the brief must convey (all three styles carry all of them):
  keep rows where status == 200 AND action in {GET, PUT} AND user does not
  start with "svc_"; sum bytes per user; top 5 by sum desc, ties by user asc;
  report.tsv = header "user<TAB>bytes" + 5 rows; summary.json = exactly
  {rows_read, rows_kept, top_user}; append "<run> ok <rows_kept>" to log.txt.
"""
import argparse, csv, hashlib, io, json, random
from pathlib import Path

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
    def ntok(s): return len(_ENC.encode(s))
except Exception:
    def ntok(s): return None

STYLES = ["prose", "telegraphic", "structured"]
MODELS = ["haiku", "sonnet"]
USERS = ["ada", "bob", "cy", "dee", "eve", "finn", "gus", "hal", "ivy", "jo",
         "svc_backup", "svc_cron", "Svc_ops"]   # Svc_ops: capital S -> NOT excluded
ACTIONS = ["GET", "PUT", "POST", "DELETE", "get"]  # lowercase get -> NOT counted
STATUSES = [200, 200, 200, 201, 404, 500]

def make_events(rng):
    rows = []
    for i in range(300):
        rows.append({
            "ts": f"2026-09-06T{i//60:02d}:{i%60:02d}:00Z",
            "user": rng.choice(USERS),
            "action": rng.choice(ACTIONS),
            "bytes": rng.randint(1, 5000),
            "status": rng.choice(STATUSES),
        })
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["ts", "user", "action", "bytes", "status"])
    w.writeheader(); w.writerows(rows)
    return buf.getvalue(), rows

def gold(rows):
    kept = [r for r in rows if r["status"] == 200 and r["action"] in ("GET", "PUT")
            and not r["user"].startswith("svc_")]
    sums = {}
    for r in kept:
        sums[r["user"]] = sums.get(r["user"], 0) + r["bytes"]
    top = sorted(sums.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {"rows_read": len(rows), "rows_kept": len(kept), "top": top,
            "top_user": top[0][0]}

def briefs(run, d):
    ev, rep, summ, log = f"{d}/events.csv", f"{d}/report.tsv", f"{d}/summary.json", f"{d}/log.txt"
    prose = (
        f"I need a small usage report built from {ev}, which is a CSV with a header row "
        f"and the columns ts, user, action, bytes, status. Please keep only the rows where "
        f"status is 200, action is exactly GET or PUT (case-sensitive), and the user name "
        f"does not start with the lowercase prefix \"svc_\". For those rows, sum bytes per "
        f"user, then take the top 5 users by that sum, descending, breaking ties by user name "
        f"ascending. Write them to {rep} as tab-separated text: a header line \"user\\tbytes\" "
        f"followed by exactly five rows. Then write {summ} as JSON with exactly three keys: "
        f"\"rows_read\" (the number of data rows in the CSV, not counting the header), "
        f"\"rows_kept\" (how many rows survived the filter), and \"top_user\" (the first user "
        f"in the report). After that, append exactly one line reading \"{run} ok N\" to the "
        f"existing file {log}, where N is rows_kept. Please don't modify the CSV and don't "
        f"create any other files. When you're finished, reply with exactly \"OK N\" (same N) "
        f"and nothing else."
    )
    telegraphic = (
        f"Usagereport from {ev} (csv hdr ts,user,action,bytes,status). Keep status200 + "
        f"action GET/PUT casesens + user !startswith lc \"svc_\". Sumbytes peruser, top5 desc, "
        f"ties userasc. Emit {rep} TSV hdr \"user\\tbytes\" + exactly5rows. Emit {summ} "
        f"exactly3keys rows_read (datarows exclhdr), rows_kept (postfilter), top_user (row1). "
        f"Append1line \"{run} ok N\" existing {log}, N=rows_kept. CSVreadonly, nootherfiles. "
        f"Reply exactly \"OK N\" only."
    )
    structured = (
        f"Task: usage report.\n"
        f"- Input: {ev} (CSV, header ts,user,action,bytes,status; read-only)\n"
        f"- Keep rows where: status == 200; action is exactly GET or PUT (case-sensitive); "
        f"user does not start with lowercase \"svc_\"\n"
        f"- Sum bytes per user; top 5 by sum desc, ties by user asc\n"
        f"- Write {rep}: TSV, header line \"user\\tbytes\", then exactly 5 rows\n"
        f"- Write {summ}: JSON with exactly 3 keys: rows_read (data rows, excl header), "
        f"rows_kept (after filter), top_user (first report row)\n"
        f"- Append exactly one line \"{run} ok N\" to existing {log}, N = rows_kept\n"
        f"- Create no other files\n"
        f"- Final reply: exactly \"OK N\", nothing else"
    )
    return {"prose": prose, "telegraphic": telegraphic, "structured": structured}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    root = Path(a.root)  # no resolve(): keep the caller-chosen (short) path in the brief
    rng = random.Random(a.seed)
    data = [make_events(rng) for _ in range(a.n)]
    out = {}
    for i in range(a.n):
        text, rows = data[i]
        g = gold(rows)
        for style in STYLES:
            for model in MODELS:
                run = "b" + hashlib.sha1(f"{a.seed}:{style}:{model}:{i}".encode()).hexdigest()[:6]
                d = root / run
                d.mkdir(parents=True, exist_ok=True)
                (d / "events.csv").write_text(text)
                (d / "log.txt").write_text("boot ok\nschema ok\nready\n")
                b = briefs(run, str(d))[style]
                out[run] = {"cell": f"{style}_{model}_{i:02d}", "style": style, "model": model,
                            "i": i, "dir": str(d), "brief": b, "brief_chars": len(b),
                            "brief_o200k": ntok(b),
                            "gold": {**g, "sha256": hashlib.sha256(text.encode()).hexdigest()}}
    (root / "briefs.json").write_text(json.dumps(out, indent=1))
    by = {}
    for r in out.values():
        by.setdefault(r["style"], []).append((r["brief_chars"], r["brief_o200k"]))
    for s, v in by.items():
        print(s, "chars", sum(c for c, _ in v) / len(v), "o200k", sum(t or 0 for _, t in v) / len(v))
    print("rows_kept", [gold(r)["rows_kept"] for _, r in data])

if __name__ == "__main__":
    main()
