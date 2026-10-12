"""Swarm main loop: baseline -> queue -> workers -> merge gate -> repeat.

    python3 swarm/run.py run [--max-workers 100] [--concurrency 30] [--max-usd 50] [--model haiku]
                             [--fake] [--batch 4] [--sample 150] [--full-every 25] [--suites a,b]
    python3 swarm/run.py status
    python3 swarm/run.py queue [--top 20]          # what would be dispatched next

Each worker is one `headless_fanout.py run` process with a one-task tasks.jsonl,
--cwd set to its own worktree (wt_root/<task id>, detached at main HEAD), tools
Read,Grep,Glob,Edit,Write,Bash, and write rules Edit(//<worktree>/**) and
Edit(//tmp/**). Its result JSON goes to wt_root/_results/<id>.json.

Resumable: everything lives in swarm/state/ (ledger.jsonl, known.json,
baseline.json, merges.jsonl, history.jsonl, tasks/<id>/). On restart, workers
whose process is still alive are adopted, dead ones are reaped, finished ones
are merged. --max-workers counts launches in this invocation; --max-usd is the
total spend recorded in the ledger across all invocations.
"""

from __future__ import annotations

if __name__ == "__main__":
    import _boot  # noqa: F401

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from swarm import merge
from swarm import queue as workqueue
from swarm import testrun as tr
from swarm.common import (FANOUT, Ctx, add_worktree, head, ledger, ledger_put, log, read_json,
                          read_jsonl, remove_worktree)

TOOLS = "Read,Grep,Glob,Edit,Write,Bash"
POLL_SECS = 5


# ---------------------------------------------------------------- launching

def source_size(ctx: Ctx):
    def size(test_id: str) -> int:
        suite, _, rel = test_id.partition("/")
        p = ctx.root / "tests" / "suites" / suite / rel
        try:
            return p.stat().st_size
        except OSError:
            return 10 ** 9
    return size


def task_prompt(ctx: Ctx, tid: str, item: workqueue.WorkItem, out: Path, prev_notes: list[str]) -> str:
    wt = ctx.worktree(tid)
    lines = [f"Task id: {tid}", f"Your worktree (current directory): {wt}", f"Result JSON out path: {out}",
             f"Scratch dir: /tmp/{tid}/", "",
             f"Failure cluster: {item.signature}",
             f"({item.size} failing tests share this signature)", "",
             "Representative tests to fix:"]
    for t in item.tests:
        suite = t.split("/", 1)[0]
        lines += [f"  - {t}",
                  f"    reason: {item.reasons.get(t, '')}",
                  f"    run: python3 tests/run.py --suite {suite} --filter '{t}' --jobs 1 -v"]
    others = [t for t in item.targets if t not in item.tests]
    if others:
        lines += ["", "Other members of the cluster (optional; note in test_ids_fixed any your fix also passes):",
                  "  " + " ".join(others[:12])]
    if prev_notes:
        lines += ["", "Notes from earlier workers on this cluster (their fixes were not merged):"]
        lines += [f"  * {n[:400]}" for n in prev_notes[-2:]]
    lines += ["", "Follow the procedure in your instructions. Read docs/CONTRACT.md first.",
              f"Finish by writing the result JSON to {out}."]
    return "\n".join(lines)


def launch(ctx: Ctx, a: argparse.Namespace, item: workqueue.WorkItem, seq: int) -> dict:
    tid = f"w{seq:05d}"
    base = head(ctx)
    wt = ctx.worktree(tid)
    add_worktree(ctx, wt, base)
    out = ctx.wt_root / "_results" / f"{tid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    tdir = ctx.task_dir(tid)
    prev = [r["result"]["notes"] for r in ledger(ctx).values()
            if r.get("signature") == item.signature and (r.get("result") or {}).get("notes")]
    task = {"id": tid, "prompt": task_prompt(ctx, tid, item, out, prev) + (f"\n[fake:{a.fake_mode}]" if a.fake_mode else ""),
            "out": str(out), "model": a.model, "tools": TOOLS,
            "allow": [f"Edit(/{wt}/**)", "Edit(//tmp/**)", "Bash"],
            "system_prompt": (ctx.swarm / "worker_prompt.md").read_text(), "timeout": a.timeout}
    (tdir / "task.jsonl").write_text(json.dumps(task) + "\n")
    work = tdir / "fanout"
    if a.fake:
        cmd = [sys.executable, str(ctx.swarm / "fake_worker.py"), str(tdir / "task.jsonl"),
               "--work", str(work), "--cwd", str(wt)]
    else:
        cmd = [sys.executable, str(FANOUT), "run", str(tdir / "task.jsonl"), "--work", str(work),
               "--schema", str(ctx.swarm / "worker_result.schema.json"), "--cwd", str(wt),
               "--model", a.model, "--concurrency", "1", "--retries", "0", "--timeout", str(a.timeout)]
        if a.per_task_usd:
            cmd += ["--max-usd-per-task", str(a.per_task_usd)]
        if a.effort:
            cmd += ["--effort", a.effort]
    logf = open(tdir / "runner.log", "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    rec = {"id": tid, "key": item.key, "signature": item.signature, "tests": item.tests,
           "targets": item.targets, "base": base, "status": "launched", "pid": p.pid,
           "cost_usd": 0.0, "result": None, "merged_commit": None, "reason": "",
           "t_launch": time.time(), "model": "fake" if a.fake else a.model}
    ledger_put(ctx, rec)
    log(f"LAUNCH {tid} [{item.priority}] {item.key}  tests: {', '.join(item.tests)}")
    return {"rec": rec, "proc": p}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # a zombie or a recycled pid that is not ours
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"fanout" in cmd or b"fake_worker" in cmd
    except OSError:
        return False


def reap(ctx: Ctx, rec: dict) -> dict:
    """Process finished: fold runner ledger + out JSON into the record."""
    tdir = ctx.task_dir(rec["id"])
    rows = read_jsonl(tdir / "fanout" / "ledger.jsonl")
    out = read_json(ctx.wt_root / "_results" / f"{rec['id']}.json")
    rec.update(cost_usd=round(sum(r.get("cost", 0) for r in rows), 4), t_done=time.time(),
               turns=sum(r.get("turns") or 0 for r in rows), denials=sum(r.get("denials", 0) for r in rows),
               result=out if isinstance(out, dict) else None)
    if not rec["result"]:
        why = rows[-1].get("reason") if rows else "runner produced no ledger"
        rec.update(status="failed", reason=f"no result: {why}")
    elif rec["result"].get("status") != "fixed":
        rec.update(status="failed", reason=f"worker reported {rec['result'].get('status')}")
    else:
        rec.update(status="finished")
    ledger_put(ctx, rec)
    log(f"DONE   {rec['id']} {rec['status']} ${rec['cost_usd']:.3f} {rec.get('reason', '')}")
    if rec["status"] == "failed":
        remove_worktree(ctx, ctx.worktree(rec["id"]))
    return rec


def spent(ctx: Ctx) -> float:
    return sum(r.get("cost_usd") or 0 for r in ledger(ctx).values())


# ---------------------------------------------------------------- loop

def run(ctx: Ctx, a: argparse.Namespace) -> int:
    if not (ctx.state / "baseline.json").exists() or not tr.known(ctx):
        merge.refresh(ctx, do_bisect=False)
    procs: dict[str, subprocess.Popen | None] = {}
    for r in ledger(ctx).values():            # adopt survivors of an interrupted run
        if r["status"] == "launched":
            procs[r["id"]] = None
    launched = 0
    stop_launch = False
    # Full refresh runs in a thread (it can take many minutes); launching and
    # reaping continue meanwhile, merging waits for it.
    refresher: threading.Thread | None = None
    while True:
        led = ledger(ctx)
        for tid in list(procs):
            p = procs[tid]
            rec = led[tid]
            running = p.poll() is None if p else alive(rec.get("pid", -1))
            if running and time.time() - rec["t_launch"] > a.timeout + 300:
                log(f"KILL   {tid}: over time")
                try:
                    os.killpg(rec["pid"], signal.SIGKILL)
                    if p:
                        p.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                running = False
            if not running:
                reap(ctx, rec)
                del procs[tid]
        finished = [r for r in ledger(ctx).values() if r["status"] == "finished"]
        refreshing = refresher is not None and refresher.is_alive()
        if not refreshing:                    # the gate tree and known.json belong to refresh
            for i in range(0, len(finished), ctx.batch):
                batch = finished[i:i + ctx.batch]
                merge.merge_batch(ctx, batch)
                for r in batch:
                    remove_worktree(ctx, ctx.worktree(r["id"]))
            if (read_json(ctx.state / "counters.json", {}) or {}).get("merges_since_full", 0) >= ctx.full_every:
                refresher = threading.Thread(target=merge.refresh, args=(ctx,), daemon=True)
                refresher.start()
                refreshing = True
        if a.max_usd and spent(ctx) >= a.max_usd and not stop_launch:
            log(f"spend ${spent(ctx):.2f} reached --max-usd; no more launches")
            stop_launch = True
        if a.max_workers and launched >= a.max_workers:
            stop_launch = True
        if not stop_launch and len(procs) < a.concurrency:
            items = workqueue.build(tr.known(ctx), ledger(ctx), source_size(ctx))
            if not items and not procs and not finished and not refreshing:
                log("queue empty and nothing in flight")
                break
            seq = len(ledger(ctx)) + 1
            for item in items[:a.concurrency - len(procs)]:
                if a.max_workers and launched >= a.max_workers:
                    break
                w = launch(ctx, a, item, seq)
                procs[w["rec"]["id"]] = w["proc"]
                seq += 1
                launched += 1
        if stop_launch and not procs and not refreshing and \
                not any(r["status"] == "finished" for r in ledger(ctx).values()):
            break
        time.sleep(a.poll)
    if a.final_refresh:
        merge.refresh(ctx)
    status(ctx)
    return 0


# ---------------------------------------------------------------- status

def status(ctx: Ctx) -> int:
    led = ledger(ctx)
    by = Counter(r["status"] for r in led.values())
    cost = spent(ctx)
    merged = by.get("merged", 0)
    print(f"workers: {len(led)} total  " + "  ".join(f"{k} {v}" for k, v in sorted(by.items())))
    print(f"spend: ${cost:.2f}" + (f"  (${cost / merged:.3f} per merge)" if merged else ""))
    c = read_json(ctx.state / "counters.json", {}) or {}
    print(f"merges since last full run: {c.get('merges_since_full', 0)}")
    hist = read_jsonl(ctx.state / "history.jsonl")
    if hist:
        suites = sorted({s for h in hist for s in h["suites"]})
        print("pass counts over time:")
        print("  " + f"{'time':<9}{'commit':<11}{'merges':>7} " + " ".join(f"{s[:14]:>15}" for s in suites))
        for h in hist[-12:]:
            cells = [f"{h['suites'][s]['pass']}/{h['suites'][s]['total']}" if s in h["suites"] else "-"
                     for s in suites]
            print("  " + f"{time.strftime('%H:%M', time.localtime(h['time'])):<9}{h['commit'][:10]:<11}"
                  f"{h.get('merges_total', 0):>7} " + " ".join(f"{x:>15}" for x in cells))
    k = tr.known(ctx)
    if k:
        counts = tr.suite_counts(k)
        print("known now: " + ", ".join(f"{s} {v['pass']}/{v['total']}" for s, v in sorted(counts.items())))
    rej = Counter(r["reason"].split(":")[0] for r in led.values() if r["status"] in ("rejected", "failed"))
    if rej:
        print("top rejection/failure reasons: " + "; ".join(f"{n}x {w}" for w, n in rej.most_common(5)))
    live = [r for r in led.values() if r["status"] == "launched"]
    for r in live[:10]:
        print(f"  in flight {r['id']} {int(time.time() - r['t_launch'])}s {r['key'][:80]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status", "queue", "refresh"])
    ap.add_argument("--root", type=Path, default=Ctx().root)
    ap.add_argument("--wt-root", type=Path, default=Ctx().wt_root)
    ap.add_argument("--max-workers", type=int, default=0, help="launches in this invocation (0 = no cap)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-usd", type=float, default=0, help="stop launching at this total ledger spend")
    ap.add_argument("--per-task-usd", type=float, default=2.0)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per worker")
    ap.add_argument("--fake", action="store_true", help="use swarm/fake_worker.py instead of claude")
    ap.add_argument("--fake-mode", default="", help="force a fake_worker mode on every task")
    ap.add_argument("--jobs", type=int, default=4, help="test parallelism in gate/baseline")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--full-every", type=int, default=25)
    ap.add_argument("--final-refresh", action="store_true", help="full run after the loop ends")
    ap.add_argument("--suites", default="", help="comma list; default every dir in tests/suites")
    ap.add_argument("--poll", type=float, default=POLL_SECS)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    ctx = Ctx(root=a.root.resolve(), wt_root=a.wt_root, jobs=a.jobs, batch=a.batch,
              sample_size=a.sample, full_every=a.full_every,
              suites=[s for s in a.suites.split(",") if s])
    if a.cmd == "status":
        return status(ctx)
    if a.cmd == "refresh":
        merge.refresh(ctx)
        return 0
    if a.cmd == "queue":
        for w in workqueue.build(tr.known(ctx), ledger(ctx), source_size(ctx), limit=a.top):
            print(f"{w.priority:7.2f} n={w.size:<4} {w.key}\n         e.g. {', '.join(w.tests)}")
        return 0
    return run(ctx, a)


if __name__ == "__main__":
    sys.exit(main())
