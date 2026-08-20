#!/usr/bin/env python3
"""Score the KV-window ladder, one subprocess per arm.

Weights are held at the shipped spec so the contrast is the window alone, plus
one combined arm at `default=2` — the configuration a device build would
actually ship, given `needle-quantization` found that spec free.
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
SCRATCH = Path(os.environ.get("KV_SCRATCH", "/tmp"))
SHIPPED = "embedding=4,mhc=4,default=2"

ARMS = [
    ("kv160", 160, SHIPPED), ("kv192", 192, SHIPPED), ("kv224", 224, SHIPPED),
    ("kv256", 256, SHIPPED), ("kv384", 384, SHIPPED), ("kv512", 512, SHIPPED),
    ("kv160-w2", 160, "default=2"),
]


def run(name: str, window: int, spec: str) -> None:
    cact = SCRATCH / f"kv_{name}.cact"
    b = subprocess.run([sys.executable, str(HERE / "build_kv.py"), "--kv-window", str(window),
                        "--spec", spec, "--out", str(cact)],
                       cwd=SCRATCH, capture_output=True, text=True)
    if b.returncode:
        print(f"  !! export failed for {name}: {b.stderr.strip().splitlines()[-1:]}", flush=True)
        return
    info = json.loads(b.stdout.strip().splitlines()[-1])
    (HERE / f"size_{name}.json").write_text(json.dumps({"arm": name, **info}, indent=1) + "\n")
    print(f"{name:10} window {window:4}  {info['mb']:6.2f} MB  {spec}", flush=True)
    subprocess.run([sys.executable, str(PRUNING / "eval_pruned.py"), "--weights", str(cact),
                    "--label", f"kv_{name}", "--cut", f"kv_window={window} {spec}",
                    "--layers", "27", "--engram", "(2, 15)", "--out-dir", str(HERE)],
                   cwd=SCRATCH)
    cact.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[a[0] for a in ARMS])
    a = ap.parse_args()
    for name, window, spec in ARMS:
        if name in a.arms:
            t0 = time.perf_counter()
            run(name, window, spec)
            print(f"     ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
