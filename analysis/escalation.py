"""Haiku vs Sonnet on the hcc failing tail: same clusters, prompt, budget and base commit.

    python3 analysis/escalation.py select --n 40 --out analysis/escalation_set.json   # on the hub
    python3 analysis/escalation.py run   --set analysis/escalation_set.json --work runs/esc --concurrency 8
    python3 analysis/escalation.py report --work runs/esc

`select` samples failing clusters from swarm/state/known.json, stratified by suite, and pins the base
commit. `run` launches every (cluster, arm, replicate) as one headless worker in its own worktree at
that base, then grades each patch in isolation at the base: SOLVED = at least one representative test
flips to PASS and no test in smoke + a fixed sample of known-passing tests regresses. Nothing is merged.
Resumable: finished (cluster, arm, rep) results are kept in <work>/results.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from swarm import _boot  # noqa: F401,E402
from swarm import queue as workqueue  # noqa: E402
from swarm.common import FANOUT  # noqa: E402

ARMS = {"haiku": {"model": "haiku", "effort": None}, "sonnet": {"model": "sonnet", "effort": "medium"}}
CHEAP = ("smoke", "c-testsuite", "chibicc", "gcc-torture", "gcc-compile", "gcc-dg", "gcc-dg-compile",
         "reject-gcc-dg", "llvm-single", "csmith", "doom", "quickjs")
TIMEOUT, USD_CAP, SAMPLE = 1500, 3.0, 120


def sh(*a, cwd=ROOT, check=True, **kw):
    return subprocess.run(list(a), cwd=cwd, check=check, capture_output=True, text=True,
                          errors="surrogateescape", **kw)


def run_tests(tree: Path, ids: list[str]) -> dict:
    if not ids:
        return {}
    out = tree / ".esc_tests.json"
    suites = sorted({t.split("/", 1)[0] for t in ids})
    cmd = [sys.executable, "tests/run.py", "--suite", ",".join(suites), "--jobs", "2", "--quiet", "--json", str(out)]
    for t in ids:
        cmd += ["--filter", t.replace("[", "[[]")]
    env = dict(os.environ, PYTHONPATH=str(tree), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=3600)
    res = json.loads(out.read_text()) if out.exists() else {}
    out.unlink(missing_ok=True)
    return res


def is_pass(r):
    return bool(r) and r.get("status") == "PASS"


# ---------------------------------------------------------------- select

def select(a):
    known = json.loads((ROOT / "swarm" / "state" / "known.json").read_text())
    clusters = workqueue.clusters({t: r for t, r in known.items() if t.split("/", 1)[0] in CHEAP})
    by_suite = collections.defaultdict(list)
    for sig, tids in clusters.items():
        by_suite[tids[0].split("/", 1)[0]].append((sig, sorted(tids)))
    rng = random.Random(20261012)
    for v in by_suite.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < a.n and any(by_suite.values()):        # round-robin across suites
        for s in sorted(by_suite):
            if by_suite[s] and len(picked) < a.n:
                picked.append(by_suite[s].pop())
    passing = sorted(t for t, r in known.items() if is_pass(r) and t.split("/", 1)[0] in ("smoke", "c-testsuite", "chibicc", "gcc-torture"))
    sample = sorted(rng.sample(passing, min(SAMPLE, len(passing))))
    base = sh("git", "rev-parse", "HEAD").stdout.strip()
    items = [{"key": f"c{i:02d}", "signature": sig, "tests": tids[:3], "members": tids[:20],
              "reasons": {t: known[t].get("reason", "") for t in tids[:3]}} for i, (sig, tids) in enumerate(picked)]
    Path(a.out).write_text(json.dumps({"base": base, "items": items, "sample": sample}, indent=1))
    print(f"{len(items)} clusters at {base[:10]}:", collections.Counter(i["tests"][0].split("/")[0] for i in items))


# ---------------------------------------------------------------- run

def prompt(item, wt: Path, out: Path, tid: str) -> str:
    lines = [f"Task id: {tid}", f"Your worktree (current directory): {wt}", f"Result JSON out path: {out}",
             f"Scratch dir: /tmp/{tid}/", "", f"Failure cluster: {item['signature']}", "",
             "Representative tests to fix:"]
    for t in item["tests"]:
        lines += [f"  - {t}", f"    reason: {item['reasons'].get(t, '')}",
                  f"    run: python3 tests/run.py --suite {t.split('/', 1)[0]} --filter '{t}' --jobs 1 -v"]
    others = [t for t in item["members"] if t not in item["tests"]]
    if others:
        lines += ["", "Other members of the cluster (optional):", "  " + " ".join(others[:12])]
    lines += ["", "Follow the procedure in your instructions. Read docs/CONTRACT.md first.",
              f"Finish by writing the result JSON to {out}."]
    return "\n".join(lines)


def one(job, base, sample, work: Path):
    item, arm, rep = job
    tid = f"esc-{item['key']}-{arm}-{rep}"
    wt = work / "wt" / tid
    out = work / "out" / f"{tid}.json"
    tdir = work / "tasks" / tid
    for p in (wt,):
        if p.exists():
            sh("git", "worktree", "remove", "--force", str(p), check=False)
    sh("git", "worktree", "add", "--detach", "--force", str(wt), base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    task = {"id": tid, "prompt": prompt(item, wt, out, tid), "out": str(out), "model": ARMS[arm]["model"],
            "tools": "Read,Grep,Glob,Edit,Write,Bash", "allow": [f"Edit(/{wt}/**)", "Edit(//tmp/**)", "Bash"],
            "system_prompt": (ROOT / "swarm" / "worker_prompt.md").read_text(), "timeout": TIMEOUT}
    (tdir / "task.jsonl").write_text(json.dumps(task) + "\n")
    cmd = [sys.executable, str(FANOUT), "run", str(tdir / "task.jsonl"), "--work", str(tdir / "fanout"),
           "--schema", str(ROOT / "swarm" / "worker_result.schema.json"), "--cwd", str(wt),
           "--model", ARMS[arm]["model"], "--concurrency", "1", "--retries", "0", "--timeout", str(TIMEOUT),
           "--max-usd-per-task", str(USD_CAP)]
    if ARMS[arm]["effort"]:
        cmd += ["--effort", ARMS[arm]["effort"]]
    t0 = time.time()
    subprocess.run(cmd, capture_output=True, text=True)
    secs = time.time() - t0
    led = {}
    for line in (tdir / "fanout" / "ledger.jsonl").open() if (tdir / "fanout" / "ledger.jsonl").exists() else []:
        led = json.loads(line)
    result = json.loads(out.read_text()) if out.exists() else None
    sh("git", "add", "-A", cwd=wt, check=False)
    patch = sh("git", "diff", "--cached", "--no-renames", "--binary", base, "--", "hcc", cwd=wt, check=False).stdout
    shutil.rmtree(f"/tmp/{tid}", ignore_errors=True)
    rec = {"key": item["key"], "arm": arm, "rep": rep, "signature": item["signature"], "secs": round(secs),
           "cost": led.get("cost"), "turns": led.get("turns"), "out_tok": led.get("out_tok"),
           "reported": (result or {}).get("status"), "patch_lines": patch.count("\n")}
    # grade on a clean tree: base + only the hcc/ patch (drops any edit the worker made elsewhere)
    sh("git", "reset", "-q", "--hard", base, cwd=wt, check=False)
    sh("git", "clean", "-fdq", cwd=wt, check=False)
    applied = patch.strip() and sh("git", "apply", "--binary", "-", cwd=wt, check=False, input=patch).returncode == 0
    if applied:
        res = run_tests(wt, item["members"] + sample)
        smoke = run_tests(wt, ["smoke/*"])
        res.update(smoke)
        flipped = [t for t in item["members"] if is_pass(res.get(t))]
        regress = [t for t in sample if t in res and not is_pass(res[t])] + \
                  [t for t, r in smoke.items() if not is_pass(r)]
        rec.update(flipped=len(flipped), reps_flipped=sum(t in flipped for t in item["tests"]),
                   regressions=regress[:10], n_regressions=len(regress))
    else:
        rec.update(flipped=0, reps_flipped=0, regressions=[], n_regressions=0)
    rec["solved"] = rec["reps_flipped"] > 0 and rec["n_regressions"] == 0
    (work / "patches").mkdir(exist_ok=True)
    (work / "patches" / f"{tid}.diff").write_text(patch, errors="surrogateescape")
    sh("git", "worktree", "remove", "--force", str(wt), check=False)
    return rec


def run(a):
    spec = json.loads(Path(a.set).read_text())
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    done_path = work / "results.jsonl"
    done = {(r["key"], r["arm"], r["rep"]) for r in map(json.loads, done_path.open())} if done_path.exists() else set()
    # Baseline at the pinned base: keep clusters whose representatives all still fail there, and
    # sample tests that pass there (known.json can be stale by a few merges).
    bpath = work / "base.json"
    if not bpath.exists():
        bt = work / "wt" / "_base"
        sh("git", "worktree", "add", "--detach", "--force", str(bt), spec["base"], check=False)
        ids = [t for i in spec["items"] for t in i["members"]] + spec["sample"]
        res = run_tests(bt, ids)
        res.update(run_tests(bt, ["smoke/*"]))
        bpath.write_text(json.dumps(res))
        sh("git", "worktree", "remove", "--force", str(bt), check=False)
    bres = json.loads(bpath.read_text())
    items = [i for i in spec["items"] if not any(is_pass(bres.get(t)) for t in i["tests"])]
    sample = [t for t in spec["sample"] if is_pass(bres.get(t))]
    for i in items:
        i["members"] = [t for t in i["members"] if not is_pass(bres.get(t))]
    print(f"base check: {len(items)}/{len(spec['items'])} clusters still failing, {len(sample)} sample tests pass", flush=True)
    jobs = [(i, arm, rep) for i in items for rep in range(a.reps) for arm in ARMS
            if (i["key"], arm, rep) not in done]
    print(f"{len(jobs)} jobs to run, {len(done)} done", flush=True)
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(one, j, spec["base"], sample, work): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:  # keep going; the job is retried on the next run
                print("job failed:", futs[f][0]["key"], futs[f][1], futs[f][2], repr(e)[:200], flush=True)
                continue
            with done_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{rec['key']} {rec['arm']}#{rec['rep']}: solved={rec['solved']} flipped={rec['flipped']} "
                  f"regr={rec['n_regressions']} ${rec['cost'] or 0:.3f} {rec['secs']}s", flush=True)


def report(a):
    rows = [json.loads(l) for l in (Path(a.work) / "results.jsonl").open()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    print("| arm | runs | solved | solve rate | clusters solved (any rep) | $ total | $ per solve | median turns | median s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, rs in by.items():
        solved = sum(r["solved"] for r in rs)
        cost = sum(r["cost"] or 0 for r in rs)
        keys = {r["key"] for r in rs if r["solved"]}
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
        print(f"| {arm} | {len(rs)} | {solved} | {solved / len(rs):.0%} | {len(keys)} | {cost:.2f} | "
              f"{cost / solved if solved else float('nan'):.3f} | {med([r['turns'] or 0 for r in rs])} | {med([r['secs'] for r in rs])} |")
    ks = {r["key"] for r in rows}
    h = {r["key"] for r in by.get("haiku", []) if r["solved"]}
    s = {r["key"] for r in by.get("sonnet", []) if r["solved"]}
    print(f"\nclusters: {len(ks)}; both solved {len(h & s)}, only haiku {len(h - s)}, only sonnet {len(s - h)}, "
          f"neither {len(ks - h - s)}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select"); p.add_argument("--n", type=int, default=40); p.add_argument("--out", required=True)
    p = sp.add_parser("run"); p.add_argument("--set", required=True); p.add_argument("--work", required=True)
    p.add_argument("--concurrency", type=int, default=8); p.add_argument("--reps", type=int, default=2)
    p = sp.add_parser("report"); p.add_argument("--work", required=True)
    a = ap.parse_args()
    {"select": select, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
