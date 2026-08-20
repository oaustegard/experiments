#!/usr/bin/env python3
"""Drive every naming variant in its own process, flat and oracle.

    python3 run_all.py                    # all 12 runs
    python3 run_all.py --modes flat       # the 6 that answer H1

A subprocess per run because the Needle engine holds one global session: a
second `needle.Needle` in the same process re-runs `needle_init`, and the
catalogue a run measures should be the only one that process ever loaded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from names import VARIANTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--modes", nargs="*", default=["flat", "oracle"])
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()

    failures = []
    for mode in a.modes:
        for v in a.variants:
            t0 = time.perf_counter()
            r = subprocess.run(
                [sys.executable, str(HERE / "eval_names.py"),
                 "--variant", v, "--mode", mode, "--repeat", str(a.repeat)],
                cwd=HERE,
            )
            if r.returncode:
                failures.append((v, mode))
                print(f"  !! {v} {mode} exited {r.returncode}", flush=True)
            else:
                print(f"     ({time.perf_counter() - t0:.0f}s)", flush=True)
    if failures:
        print(f"failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
