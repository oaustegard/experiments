"""Check every task's environment: the gold patch must resolve it and the empty
patch must not. Only tasks passing both enter the pool.

    python harness/validate.py [--workers 3]   -> data/validation.jsonl (append, resumable)

One line per (instance, arm). A task whose FAIL_TO_PASS tests already pass at
base here would count a no-op patch as a fix, so it is dropped too.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_tasks  # noqa: E402
from grade import grade  # noqa: E402

OUT = DATA / "validation.jsonl"
LOGS = DATA / "logs" / "validation"
_lock = threading.Lock()


def done() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    return {(r["instance_id"], r["arm"]) for r in map(json.loads, OUT.open())}


def one(task: dict, arm: str) -> None:
    patch = task["patch"] if arm == "gold" else ""
    try:
        r = grade(task, patch, "validate", f"{task['instance_id']}--{arm}",
                  log_path=LOGS / f"{task['instance_id']}--{arm}.log")
    except Exception as e:  # noqa: BLE001 - record and move on
        r = {"resolved": None, "error": f"{type(e).__name__}: {e}"[:500]}
    r = {"instance_id": task["instance_id"], "arm": arm, **r}
    with _lock, OUT.open("a") as f:
        f.write(json.dumps(r) + "\n")
    print(arm, task["instance_id"], r.get("resolved"), r.get("seconds"), r.get("error", ""), flush=True)


def pool() -> list[str]:
    """Instance ids whose gold resolves and whose base does not.

    Verdicts are re-scored from the saved logs with the current parser, so a
    parser fix applies without rewriting validation.jsonl."""
    from grade import report, status_map
    tasks = load_tasks()
    rows = {}
    for r in map(json.loads, OUT.open()):
        log = LOGS / f"{r['instance_id']}--{r['arm']}.log"
        if not r.get("error") and log.exists():
            t = tasks[r["instance_id"]]
            r.update(report(t, status_map(t, log.read_text())))
        rows[(r["instance_id"], r["arm"])] = r
    ids = {i for i, _ in rows}
    return sorted(i for i in ids
                  if rows.get((i, "gold"), {}).get("resolved") is True
                  and rows.get((i, "base"), {}).get("resolved") is False
                  and not rows.get((i, "base"), {}).get("error"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    tasks = load_tasks()
    seen = done()
    jobs = [(t, arm) for t in tasks.values() for arm in ("gold", "base") if (t["instance_id"], arm) not in seen]
    print(f"{len(jobs)} runs to go", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: one(*j), jobs))
    p = pool()
    (DATA / "pool.json").write_text(json.dumps(p, indent=0))
    print(f"POOL {len(p)} of {len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
