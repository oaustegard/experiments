#!/usr/bin/env python3
"""Run worker attempts as headless `claude -p` children and capture what a judge needs.

One background job, checkpointed per attempt (an existing data/attempts/<arm>_r<k>_<task>.json is
skipped). Per attempt it stores the issue text, the final diff against the pristine task, the
worker's final summary, the visible-test output, the held-out result, token usage per model call
(peak context = max input+cache read+cache write over calls), and cost.

Headless recipe from METHODS.md ("Headless recipe"): scratch cwd, no project settings or MCP,
prompt on stdin because `--tools` swallows a positional prompt. The prompt is emit_prompts.PROMPT,
unchanged.

Usage:
  python3 run_workers.py --arms haiku sonnet-low opus-high --reps 1 2 [--tasks a b] [--jobs 6] [--limit N]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "harness"))
import emit_prompts  # noqa: E402

TASKS = EXP / "tasks"
OUT = HERE / "data" / "attempts"
ARMS = {  # arm -> (pinned model id, effort or None)
    "haiku": ("claude-haiku-5-5", None),
    "sonnet-low": ("claude-sonnet-5-5", "low"),
    "opus-high": ("claude-opus-5-5", "high"),
}
EXCLUDE = ["__pycache__", ".pytest_cache", "issue.md", "*.pyc"]


def all_tasks():
    return sorted(p.name for p in TASKS.iterdir() if (p / "meta.json").exists())


def diff_against_pristine(task, wt):
    cmd = ["diff", "-ruN"] + [a for e in EXCLUDE for a in ("-x", e)] + [str(TASKS / task / "repo"), str(wt)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_stream(text):
    seen, peak, calls, result = set(), 0, 0, {}
    for line in text.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "result":
            result = d
        m = d.get("message")
        if d.get("type") == "assistant" and isinstance(m, dict) and m.get("id") not in seen:
            seen.add(m.get("id"))
            u = m.get("usage") or {}
            peak = max(peak, u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                       + u.get("cache_creation_input_tokens", 0))
            calls += 1
    return peak, calls, result


def grade(arm_key, task, wt, scratch):
    man = scratch / f"{arm_key}_{task}.json"
    out = scratch / f"{arm_key}_{task}.graded.json"
    man.write_text(json.dumps({arm_key: {task: str(wt)}}))
    subprocess.run([sys.executable, str(EXP / "harness" / "grade_agentic.py"), "--runs", str(man),
                    "--out", str(out)], capture_output=True, text=True)
    return json.loads(out.read_text())[arm_key][task]


def run_one(arm, rep, task, root, scratch, timeout):
    dest = OUT / f"{arm}_r{rep}_{task}.json"
    if dest.exists():
        return task, "skip"
    model, effort = ARMS[arm]
    run_root = Path(root) / f"{arm}_r{rep}"
    wt = emit_prompts.worktree_for(run_root, task)
    shutil.rmtree(wt, ignore_errors=True)
    wt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TASKS / task / "repo", wt)
    shutil.copy(TASKS / task / "issue.md", wt / "issue.md")
    prompt = emit_prompts.prompt_for(run_root, task)
    cmd = ["claude", "-p", "--model", model] + (["--effort", effort] if effort else []) + [
        "--permission-mode", "bypassPermissions", "--setting-sources", "", "--strict-mcp-config",
        "--no-session-persistence", "--output-format", "stream-json", "--verbose",
        "--tools", "Bash", "Read", "Edit", "Write", "Glob", "Grep"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=scratch)
        stream, err = p.stdout, p.stderr[-500:]
    except subprocess.TimeoutExpired as e:
        stream, err = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
    peak, calls, res = parse_stream(stream)
    (HERE / "data" / "streams").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "streams" / f"{arm}_r{rep}_{task}.jsonl").write_text(stream)
    vis = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=wt, capture_output=True,
                         text=True).stdout[-1500:]
    g = grade(f"{arm}_r{rep}", task, wt, scratch)
    rec = dict(arm=arm, model=model, effort=effort, task=task, replicate=rep,
               issue=(wt / "issue.md").read_text(), diff=diff_against_pristine(task, wt),
               summary=res.get("result"), visible_output=vis,
               visible_green=("passed" in vis and "failed" not in vis and "error" not in vis.lower()),
               heldout_pass=g["passed"], heldout=f"{g['n_passed']}/{g['n_total']}",
               tampered_tests=g["tampered_tests"], edits_outside_package=g["edits_outside_package"],
               peak_ctx=peak, model_calls=calls, num_turns=res.get("num_turns"),
               cost_usd=res.get("total_cost_usd"), model_usage=res.get("modelUsage"),
               is_error=res.get("is_error"), wall_s=round(time.time() - t0, 1), stderr=err)
    OUT.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return task, f"{'PASS' if rec['heldout_pass'] else 'fail'} ${rec['cost_usd']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--root", default="/home/user/runs/jev")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="jev-scratch-"))
    plan = [(arm, r, t) for r in a.reps for arm in a.arms for t in (a.tasks or all_tasks())]
    if a.limit:
        plan = plan[:a.limit]
    print(f"{len(plan)} attempts, {a.jobs} parallel", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [(x, ex.submit(run_one, *x, a.root, scratch, a.timeout)) for x in plan]
        for x, f in futs:
            try:
                print(x, f.result(), flush=True)
            except Exception as e:  # keep going; the attempt is simply not checkpointed
                print(x, "ERROR", repr(e), flush=True)


if __name__ == "__main__":
    main()
