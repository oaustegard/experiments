#!/usr/bin/env python3
"""Export and score every bit-width arm, one subprocess each.

    python3 run_quant.py                 # all arms
    python3 run_quant.py --arms shipped all4

A subprocess per arm because the engine cannot unload a loaded `.cact`
(`needle-bsky/RESULTS.md`), and the blob is deleted after scoring so the sweep
needs ~14 MB of disk rather than 10x that. Scoring is
`needle-layer-pruning/eval_pruned.py`, imported by path rather than copied, so
every number lands on the same 62 queries and the same scorer as the pruning
sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

PRUNING = experiment("needle-layer-pruning")
SCRATCH = Path(os.environ.get("QUANT_SCRATCH", "/tmp"))

ARMS = {
    "shipped":     "embedding=4,mhc=4,default=2",
    "all4":        "default=4",
    "all3":        "default=3",
    "all2":        "default=2",
    "prot3":       "embedding=4,mhc=4,default=3",
    "prot-tern":   "embedding=4,mhc=4,default=1.58",
    "all-tern":    "default=1.58",
    "emb2":        "embedding=2,mhc=4,default=2",
    "mhc2":        "embedding=4,mhc=2,default=2",
    "engram-tern": "embedding=4,mhc=4,engram0.tables=1.58,engram1.tables=1.58,default=2",
}


def run(name: str, spec: str) -> None:
    cact = SCRATCH / f"q_{name}.cact"
    build = subprocess.run(
        [sys.executable, str(HERE / "build_quant.py"), "--spec", spec, "--out", str(cact)],
        cwd=SCRATCH, capture_output=True, text=True)
    if build.returncode:
        print(f"  !! export failed for {name}: {build.stderr.strip().splitlines()[-1:]}", flush=True)
        return
    info = json.loads(build.stdout.strip().splitlines()[-1])
    (HERE / f"size_{name}.json").write_text(json.dumps({"arm": name, **info}, indent=1) + "\n")
    print(f"{name:12} {info['mb']:6.2f} MB  {info['tensors']} tensors  {spec}", flush=True)

    subprocess.run(
        [sys.executable, str(PRUNING / "eval_pruned.py"), "--weights", str(cact),
         "--label", f"quant_{name}", "--cut", spec, "--layers", "27",
         "--engram", "(2, 15)", "--out-dir", str(HERE)],
        cwd=SCRATCH)
    cact.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    for name in a.arms:
        t0 = time.perf_counter()
        run(name, ARMS[name])
        print(f"     ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
