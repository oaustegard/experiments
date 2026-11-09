"""Run the whole schedule, one run at a time, resumable.

Each run is a child process, so a crash in one run does not end the schedule.
A run with a DONE marker is skipped; an unfinished swarm run resumes from its
archive and ledger; an unfinished Arm A run restarts clean (arm_a.py).

Before each run the subscription's rate-limit windows are read from a
one-token call. Above FIVE_HOUR_WAIT the schedule sleeps until that window
resets; above WEEK_STOP it stops, so the experiment cannot eat the week.

Completion line: 'schedule done -> results/summary.json'. A background task
is stopped after two hours, so the schedule stops itself before a run that
would not finish inside BUDGET_MIN and prints 'schedule paused -> relaunch';
launching it again picks up where it stopped.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import JAIL, run_dir, setup_jail
from gateway import Gateway

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CAP = 3.0
WALL = 3600
FIVE_HOUR_WAIT = 0.75
WEEK_STOP = 0.40
BUDGET_MIN = 112

MAIN = []
for s in (1, 2, 3):
    MAIN += [(f"A-s{s}", ["arm_a.py", f"A-s{s}", "--seed", str(s)]),
             (f"B-s{s}", ["swarm.py", f"B-s{s}", "--arm", "B", "--seed", str(s), "--K", "4", "--N", "8", "--synth", "haiku"]),
             (f"C-s{s}", ["swarm.py", f"C-s{s}", "--arm", "C", "--seed", str(s), "--K", "4", "--N", "8", "--synth", "sonnet"])]
# Exploratory K x N sweep: one seed, $1 cap, 30 minutes. Not part of the A-vs-B test.
TUNE = [(f"T-K{k}N{n}", ["swarm.py", f"T-K{k}N{n}", "--arm", f"T", "--seed", "11", "--K", str(k), "--N", str(n),
                         "--synth", "haiku", "--cap", "1.0", "--wall", "1800"])
        for k, n in ((4, 8), (4, 0), (8, 4), (2, 16))]
# Phase 2 (2026-11-09, Oskar: smaller reasoning budgets, harder tasks).
# L = the swarm at the smallest budget claude -p allows: proposers and the
# synthesizer both at effort low (Haiku 5.5's thinking cannot be switched off or
# capped through the CLI; measured). Harder tasks: Heilbronn triangle n=11 and
# hexagon packing n=11, OpenEvolve's AlphaEvolve examples.
LOW = ["--prop-effort", "low", "--synth-effort", "low"]
PHASE2 = []
for prob, tag in (("heilbronn", "ht"), ("hexagon", "hx")):
    for s in (1, 2, 3):
        PHASE2 += [(f"{tag}-A-s{s}", ["arm_a.py", f"{tag}-A-s{s}", "--seed", str(s), "--problem", prob]),
                   (f"{tag}-B-s{s}", ["swarm.py", f"{tag}-B-s{s}", "--arm", "B", "--seed", str(s), "--problem", prob]),
                   (f"{tag}-L-s{s}", ["swarm.py", f"{tag}-L-s{s}", "--arm", "L", "--seed", str(s), "--problem", prob, *LOW])]
    if prob == "heilbronn":
        PHASE2 += [(f"L-s{s}", ["swarm.py", f"L-s{s}", "--arm", "L", "--seed", str(s), *LOW]) for s in (1, 2, 3)]
SCHEDULE = MAIN + PHASE2 + TUNE


def est_min(run_id: str, args: list) -> int:
    """Worst-case minutes for a run, so the schedule pauses before the 2 h limit."""
    if run_id.startswith("T-"):
        return 33
    if run_id.startswith("A-"):
        return 15          # circle packing's Arm A hits the cap in under 10 minutes
    return 63


def quota() -> dict:
    gw = Gateway(run_dir("_quota"), cap=1.0)
    r = gw.call("haiku", ".", "Reply with the single character: .", effort="low",
                max_tokens=64, role="quota", check_budget=False)
    return r.get("quota") or {}


def collect(run_id: str):
    """Copy a run's derived records out of the jail into results/runs/<id>/."""
    src = JAIL / "runs" / run_id
    dst = RESULTS / "runs" / run_id
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("events.jsonl", "ledger.jsonl", "rounds.jsonl", "best_program.py", "config.yaml", "overhead.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    if (src / "DONE").exists():
        shutil.copy(src / "DONE", dst / "summary.json")
    best = src / "oe_out" / "best" / "best_program.py"
    if best.exists():
        shutil.copy(best, dst / "best_program.py")


def main():
    t_launch = time.time()
    setup_jail()
    calib = JAIL / "runs" / "_calib" / "overhead.json"
    if not calib.exists():
        Gateway(run_dir("_calib"), cap=1.0).calibrate()
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    for run_id, args in SCHEDULE:
        if (JAIL / "runs" / run_id / "DONE").exists():
            collect(run_id)
            continue
        est = est_min(run_id, args)
        if (time.time() - t_launch) / 60 + est > BUDGET_MIN:
            print(time.strftime("%H:%M:%S"), "schedule paused -> relaunch (next:", run_id, ")", flush=True)
            return
        while True:
            q = quota()
            print(time.strftime("%H:%M:%S"), run_id, "quota", q, flush=True)
            if (q.get("seven_day") or 0) >= WEEK_STOP:
                print("schedule stopped: seven-day utilisation", q, flush=True)
                return
            if (q.get("five_hour") or 0) < FIVE_HOUR_WAIT:
                break
            time.sleep(600)
        extra = [] if "--cap" in args else ["--cap", str(CAP), "--wall", str(WALL)]
        with open(RESULTS / "logs" / f"{run_id}.log", "a") as log:
            p = subprocess.run([sys.executable, *args, *extra], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        collect(run_id)
        print(time.strftime("%H:%M:%S"), run_id, "rc", p.returncode, flush=True)
        if p.returncode != 0:
            tail = (RESULTS / "logs" / f"{run_id}.log").read_text()[-1500:]
            print(tail, flush=True)
            if "QuotaGuard" in tail:
                print("schedule stopped: quota guard", flush=True)
                return
        if (HERE / "analyze.py").exists():
            with open(RESULTS / "logs" / "analyze.log", "a") as log:
                try:
                    subprocess.run([sys.executable, "analyze.py"], cwd=HERE, timeout=300,
                                   stdout=log, stderr=subprocess.STDOUT)
                except subprocess.TimeoutExpired:
                    print("analyze.py timed out", file=log)
    print("schedule done -> results/summary.json", flush=True)


if __name__ == "__main__":
    main()
