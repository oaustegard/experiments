"""Inside the jail: score every program in a directory with the stock evaluator.

Usage: python3 evalkit.py DIR [--timeout 60] [--workers 4]
Writes DIR/results.json: {filename: {"metrics": {...}, "t": end_time}}.
Each program runs in its own process group, killed at the timeout, which is
the 60 s that Arm A's OpenEvolve config gives its evaluator.
"""
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBLEM = HERE.parent / "problem"
ONE = ("import json,sys; sys.path.insert(0, sys.argv[1]); import evaluator; "
       "m = evaluator.evaluate(sys.argv[2]); "
       "print('@@RESULT@@' + json.dumps({k: v for k, v in m.items() if isinstance(v, (int, float, str))}))")


def score(path, timeout):
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", ONE, str(PROBLEM), str(path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return {"metrics": {"combined_score": 0.0, "validity": 0.0, "sum_radii": 0.0,
                            "error": f"Timeout after {timeout}s"}, "t": time.time(), "dt": time.time() - t0}
    for line in out.splitlines():
        if line.startswith("@@RESULT@@"):
            return {"metrics": json.loads(line[10:]), "t": time.time(), "dt": time.time() - t0}
    return {"metrics": {"combined_score": 0.0, "validity": 0.0, "sum_radii": 0.0,
                        "error": ("no result: " + err[-600:]).strip()}, "t": time.time(), "dt": time.time() - t0}


def main():
    d = Path(sys.argv[1])
    timeout = float(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 60.0
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    progs = sorted(p for p in d.glob("*.py"))
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip((p.name for p in progs), ex.map(lambda p: score(p, timeout), progs)))
    tmp = d / "results.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(d / "results.json")


if __name__ == "__main__":
    main()
