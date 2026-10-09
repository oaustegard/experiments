"""Drive one wave of agents over SWE-bench tasks, wrapping claude-workspace's fanout.py.

    ladder.py stage RUN --rung 1 --model haiku --ids data/pool.json [--limit N]
    ladder.py stage RUN --rung 2 --model sonnet --from RUN1     # RUN1's unresolved tasks
    ladder.py next RUN --slots 10       # makes checkouts, prints Agent calls (JSON lines)
    ladder.py done RUN ID [ID ...]      # on each completion: save diff, drop checkout
    ladder.py grade RUN [--workers 3]   # grade every saved diff (run in background)
    ladder.py status RUN

RUN is data/runs/<name>/. A checkout exists only while its agent runs: a Django
checkout is 64 MB, so 300 of them at once would not fit. The diff is saved as
patches/<id>.diff the moment the agent reports, whether or not it wrote its
summary file; an agent's own report is not the grade.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, JAIL, WORK, load_tasks  # noqa: E402
import prompts  # noqa: E402
import workspace  # noqa: E402

sys.path.insert(0, str(JAIL.parent))
import fanout  # noqa: E402  (claude-workspace/scripts/fanout.py)

RUNS = DATA / "runs"
SCHEMA = {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}}


def rdir(run: str) -> Path:
    return RUNS / run


def meta(run: str) -> dict:
    return json.loads((rdir(run) / "meta.json").read_text())


def checkout_path(run: str, iid: str) -> Path:
    return WORK / run / iid


def results(run: str) -> dict[str, dict]:
    f = rdir(run) / "results.jsonl"
    if not f.exists():
        return {}
    return {r["instance_id"]: r for r in map(json.loads, f.open())}


def stage(a) -> None:
    d = rdir(a.run)
    if d.exists():
        sys.exit(f"{d} exists; use a fresh run name")
    if a.rung == 1:
        ids = json.loads(Path(a.ids).read_text())
    else:
        prev = results(a.from_run)
        ids = sorted(i for i, r in prev.items() if r.get("resolved") is False)
    if a.limit:
        ids = ids[:a.limit]
    tasks = load_tasks(set(ids))
    d.mkdir(parents=True)
    (d / "schema.json").write_text(json.dumps(SCHEMA))
    model_arg = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}[a.model]
    lines = []
    for iid in ids:
        lines.append(json.dumps({
            "id": iid, "prompt": "(written at dispatch)", "out": str(d / "out" / f"{iid}.json"),
            "description": f"{a.run} {iid}", "model": model_arg, "subagent_type": "general-purpose"}))
    (d / "tasks.jsonl").write_text("\n".join(lines) + "\n")
    (d / "meta.json").write_text(json.dumps({"rung": a.rung, "model": a.model, "from": a.from_run,
                                             "n": len(ids)}, indent=1))
    fanout.init(d / "fanout", d / "tasks.jsonl", d / "schema.json", retries=0)
    (d / "out").mkdir()
    print(f"staged {len(ids)} tasks in {d}")


def _prepare(run: str, m: dict, task: dict) -> str:
    """Make the checkout and return the prompt for this task."""
    iid = task["instance_id"]
    path = workspace.checkout(task, run)
    if m["rung"] == 1:
        return prompts.rung1(task, path)
    prev = m["from"]
    diff = (rdir(prev) / "patches" / f"{iid}.diff").read_text()
    ok, how = workspace.apply(path, diff)
    if not ok:
        raise RuntimeError(f"{iid}: previous diff no longer applies: {how[:300]}")
    import subprocess
    subprocess.run(["chmod", "-R", "a+rwX", str(path)], check=True)
    r = results(prev)[iid]
    failing = r.get("f2p_fail", []) + r.get("p2p_fail", [])
    log = (rdir(prev) / "logs" / f"{iid}.log")
    excerpt = prompts.failure_excerpt(task, log.read_text() if log.exists() else "", failing)
    if r.get("error"):
        excerpt = f"(grading error: {r['error']})\n" + excerpt
    return prompts.rung2(task, path, diff, failing, excerpt)


SETTLE = 600   # seconds an out file must age before next() collects it unasked


def _finished_running(run: str) -> list[str]:
    """Running tasks whose out file is at least SETTLE seconds old.

    Agents told to write the out file last sometimes keep testing afterwards; a
    fresh out file does not mean the agent stopped (r2-sonnet django-16256 lost
    its checkout mid-run to an immediate collect). Completion notices name the
    finished ids explicitly; this only catches ones that were missed.
    """
    import time
    state = fanout._load(rdir(run) / "fanout")
    tasks = state["tasks"] if isinstance(state, dict) and "tasks" in state else state
    tasks = tasks.values() if isinstance(tasks, dict) else tasks
    now = time.time()
    return [t["id"] for t in tasks if t.get("status") == "running" and Path(t["out"]).exists()
            and now - Path(t["out"]).stat().st_mtime > SETTLE]


def next_(a) -> None:
    d, m = rdir(a.run), meta(a.run)
    # fanout.next_batch marks these done itself; collect their diffs first, or a
    # sweep-promoted task's checkout would never be saved.
    finished = _finished_running(a.run)
    if finished:
        done(argparse.Namespace(run=a.run, ids=finished))
    state = fanout._load(d / "fanout")
    batch = fanout.next_batch(state, a.slots)
    tasks = load_tasks({t["id"] for t in batch})
    for t in batch:
        t["prompt"] = _prepare(a.run, m, tasks[t["id"]]).replace("{out}", t["out"])
    fanout._save(d / "fanout", state)
    for t in batch:
        print(json.dumps(fanout.agent_call(t, d / "prompts")))
    c = fanout.counts(state)
    print(f"# {c}", file=sys.stderr)


def done(a) -> None:
    d = rdir(a.run)
    state = fanout._load(d / "fanout")
    tasks = load_tasks(set(a.ids))
    (d / "patches").mkdir(exist_ok=True)
    for iid in a.ids:
        path = checkout_path(a.run, iid)
        if path.exists():
            (d / "patches" / f"{iid}.diff").write_text(
                workspace.diff(path, tasks[iid]["base_commit"]))
            workspace.remove(tasks[iid], path)
    print("\n".join(fanout.mark_done(state, a.ids)))
    fanout._save(d / "fanout", state)
    print(f"# {fanout.counts(state)}")


_lock = threading.Lock()


def grade_run(a) -> None:
    from grade import grade
    d = rdir(a.run)
    have = results(a.run)
    todo = sorted(p.stem for p in (d / "patches").glob("*.diff") if p.stem not in have)
    tasks = load_tasks(set(todo))

    def one(iid):
        try:
            r = grade(tasks[iid], (d / "patches" / f"{iid}.diff").read_text(), f"grade-{a.run}",
                      iid, log_path=d / "logs" / f"{iid}.log")
        except Exception as e:  # noqa: BLE001
            r = {"resolved": None, "error": f"{type(e).__name__}: {e}"[:500]}
        with _lock, (d / "results.jsonl").open("a") as f:
            f.write(json.dumps({"instance_id": iid, **r}) + "\n")
        print(iid, r.get("resolved"), r.get("seconds"), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    res = results(a.run)
    print(f"GRADED {len(res)}: resolved {sum(r.get('resolved') is True for r in res.values())}")


def status(a) -> None:
    d = rdir(a.run)
    state = fanout._load(d / "fanout")
    res = results(a.run)
    print(json.dumps({"fanout": fanout.counts(state), "patches": len(list((d / "patches").glob("*.diff")))
                      if (d / "patches").exists() else 0, "graded": len(res),
                      "resolved": sum(r.get("resolved") is True for r in res.values())}))


def merge(a) -> None:
    """OUT = BASE with OVERRIDE's tasks replacing BASE's (patch, log, result).

    Used when a subset of a run is redone: rung 2 stages from OUT.
    """
    import shutil
    out, base, over = rdir(a.out), rdir(a.base), rdir(a.override)
    if out.exists():
        sys.exit(f"{out} exists")
    redo = set(results(a.override))
    missing = sorted({p.stem for p in (over / "patches").glob("*.diff")} - redo)
    if missing:
        sys.exit(f"{a.override} has ungraded patches: {missing[:5]}")
    for sub in ("patches", "logs"):
        (out / sub).mkdir(parents=True)
        for src in (base, over):
            for f in (src / sub).glob("*"):
                if src is over or f.stem not in redo:
                    shutil.copyfile(f, out / sub / f.name)
    rows = {**results(a.base), **results(a.override)}
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    (out / "meta.json").write_text(json.dumps({"merged": [a.base, a.override], "n": len(rows)}, indent=1))
    print(f"{out}: {len(rows)} results, {len(redo)} from {a.override}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("run")
    s.add_argument("--rung", type=int, choices=(1, 2), required=True)
    s.add_argument("--model", choices=("haiku", "sonnet", "opus"), required=True)
    s.add_argument("--ids")
    s.add_argument("--from", dest="from_run")
    s.add_argument("--limit", type=int)
    n = sub.add_parser("next")
    n.add_argument("run")
    n.add_argument("--slots", type=int, default=10)
    dn = sub.add_parser("done")
    dn.add_argument("run")
    dn.add_argument("ids", nargs="+")
    g = sub.add_parser("grade")
    g.add_argument("run")
    g.add_argument("--workers", type=int, default=3)
    st = sub.add_parser("status")
    st.add_argument("run")
    c = sub.add_parser("cycle", help="done IDS, then next: one call per completion notice")
    c.add_argument("run")
    c.add_argument("ids", nargs="+")
    c.add_argument("--slots", type=int, default=10)
    mg = sub.add_parser("merge", help="OUT = BASE with OVERRIDE's tasks replaced")
    mg.add_argument("out")
    mg.add_argument("base")
    mg.add_argument("override")
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a)
        return
    if a.cmd == "cycle":
        done(a)
        next_(a)
        return
    {"stage": stage, "next": next_, "done": done, "grade": grade_run, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
