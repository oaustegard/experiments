"""OpenEvolve's stock circle-packing evaluator, with one line logged per call.

The scoring is untouched: each function delegates to examples/circle_packing/
evaluator.py. The log gives Arm A the same per-evaluation record (time, code
hash, metrics) that Arm B's evalkit writes, so both arms' curves come from the
same kind of data.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "problem"))
import evaluator as _stock  # noqa: E402

LOG = os.environ.get("HSP_EVAL_LOG")


def _log(fn, program_path, metrics, t0):
    if not LOG:
        return
    try:
        code = Path(program_path).read_text()
    except OSError:
        code = ""
    row = {"t": time.time(), "dt": time.time() - t0, "fn": fn,
           "sha": hashlib.sha256(code.encode()).hexdigest()[:16],
           "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}}
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def evaluate(program_path):
    t0 = time.time(); m = _stock.evaluate(program_path); _log("evaluate", program_path, m, t0); return m


def evaluate_stage1(program_path):
    t0 = time.time(); m = _stock.evaluate_stage1(program_path); _log("stage1", program_path, m, t0); return m


def evaluate_stage2(program_path):
    t0 = time.time(); m = _stock.evaluate_stage2(program_path); _log("stage2", program_path, m, t0); return m
