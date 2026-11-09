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
EST_MIN = {"A": 15, "B": 63, "C": 63, "T": 33}   # worst case per run, minutes

MAIN = []
for s in (1, 2, 3):
    MAIN += [(f"A-s{s}", ["arm_a.py", f"A-s{s}", "--seed", str(s)]),
             (f"B-s{s}", ["swarm.py", f"B-s{s}", "--arm", "B", "--seed", str(s), "--K", "4", "--N", "8", "--synth", "haiku"]),
             (f"C-s{s}", ["swarm.py", f"C-s{s}", "--arm", "C", "--seed", str(s), "--K", "4", "--N", "8", "--synth", "sonnet"])]
# Exploratory K x N sweep: one seed, $1 cap, 30 minutes. Not part of the A-vs-B test.
TUNE = [(f"T-K{k}N{n}", ["swarm.py", f"T-K{k}N{n}", "--arm", f"T", "--seed", "11", "--K", str(k), "--N", str(n),
                         "--synth", "haiku", "--cap", "1.0", "--wall", "1800"])
        for k, n in ((4, 8), (4, 0), (8, 4), (2, 16))]
SCHEDULE = MAIN + TUNE


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
        est = EST_MIN[run_id[0]]
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
