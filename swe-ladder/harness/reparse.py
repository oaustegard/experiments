"""Recompute grading verdicts from saved test logs, after a parser change.

    python harness/reparse.py data/validation.jsonl data/logs/validation [--write]

Rows whose log exists are re-scored with grade.status_map/report; rows with an
error (patch did not apply, crash) are left alone. Prints every changed verdict.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_tasks  # noqa: E402
from grade import report, status_map  # noqa: E402


def main(rows_path: str, logs: str, write: bool):
    tasks = load_tasks()
    rows = [json.loads(l) for l in open(rows_path)]
    changed = 0
    for r in rows:
        log = Path(logs) / f"{r['instance_id']}--{r['arm']}.log"
        if r.get("error") or not log.exists():
            continue
        t = tasks[r["instance_id"]]
        new = report(t, status_map(t, log.read_text()))
        if new["resolved"] != r["resolved"]:
            changed += 1
            print(r["arm"], r["instance_id"], r["resolved"], "->", new["resolved"],
                  new["f2p"], new["p2p"], new["f2p_fail"][:1], new["p2p_fail"][:2])
        r.update(new)
    print(f"{changed} verdicts changed of {len(rows)}")
    if write:
        Path(rows_path).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--write" in sys.argv)
