#!/usr/bin/env python3
"""Export and score every lane-preserving cut of a given size.

    python3 run_sweep.py --count 4              # all 24 positions of a 4-layer cut
    python3 run_sweep.py --count 8 --starts 8 12 16

Each arm is a fresh subprocess (the engine cannot unload a `.cact`) and its
blob is deleted after scoring, so the sweep needs ~14 MB of disk, not 24x that.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("PRUNE_SCRATCH", "/tmp"))
NUM_LAYERS, ENGRAM = 27, (2, 15)


def sites_after(start: int, count: int) -> tuple:
    return tuple(l - count if l >= start + count else l
                 for l in ENGRAM if not (start <= l < start + count))


def run(label: str, start, count) -> None:
    cact = SCRATCH / f"{label}.cact"
    build = [sys.executable, str(HERE / "build_pruned.py"), "--out", str(cact)]
    if count:
        build += ["--start", str(start), "--count", str(count)]
    if subprocess.run(build, cwd=SCRATCH).returncode:
        print(f"  !! export failed for {label}", flush=True)
        return
    sites = sites_after(start, count) if count else ENGRAM
    subprocess.run(
        [sys.executable, str(HERE / "eval_pruned.py"), "--weights", str(cact),
         "--label", label, "--cut", "none" if not count else f"[{start},{start + count})",
         "--layers", str(NUM_LAYERS - count), "--engram", str(sites)],
        cwd=SCRATCH,
    )
    cact.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--starts", nargs="*", type=int, default=None)
    ap.add_argument("--control", action="store_true", help="also run the unpruned export")
    a = ap.parse_args()

    if a.control:
        t0 = time.perf_counter()
        run("control", None, 0)
        print(f"     ({time.perf_counter() - t0:.0f}s)", flush=True)

    starts = a.starts if a.starts is not None else range(NUM_LAYERS - a.count + 1)
    for s in starts:
        t0 = time.perf_counter()
        run(f"cut{a.count}_at{s:02d}", s, a.count)
        print(f"     ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
