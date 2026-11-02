"""Training sweep. One unit = one fit; a unit whose output exists is skipped, so the job resumes.

    python3 run_sweep.py            # all units, in priority order
Outputs: results/runs/<image>_<arm>.json, log in results/sweep.log (stdout).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec  # noqa: E402

IMAGES = ["kodim23", "kodim05", "kodim01", "kodim07", "kodim15", "kodim20"]
RUNS = HERE / "results" / "runs"


def units():
    for im in IMAGES:                       # main result: one prefix-trained run gives a whole curve
        yield im, "prefix1024", dict(n=1024, prefix=True)
    for im in IMAGES:                       # dedicated plain runs: what prefix validity costs
        yield im, "plain64", dict(n=64, prefix=False)
        yield im, "plain256", dict(n=256, prefix=False)
    for im in IMAGES:                       # residual MLP on top of the full prefix run
        yield im, "res16", dict(residual_of="prefix1024", h=16)
    for im in IMAGES[:2]:                   # plain at full size: prefixes untrained
        yield im, "plain1024", dict(n=1024, prefix=False)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    for im, arm, kw in units():
        out = RUNS / f"{im}_{arm}.json"
        if out.exists():
            continue
        img = HERE / "data" / "kodak" / f"{im}.png"
        t0 = time.time()
        print(f"== {im} {arm} start {time.strftime('%H:%M:%S')}", flush=True)
        try:
            if "residual_of" in kw:
                src = RUNS / f"{im}_{kw['residual_of']}.json"
                run, _ = codec.load_run(src, img)
                net = codec.fit_residual(run, h=kw["h"], steps=1500)
                codec.save_run(run, out, residual=net)
            else:
                run = codec.fit(img, **kw)
                codec.save_run(run, out)
            print(f"== {im} {arm} done {time.time() - t0:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"== {im} {arm} FAILED", flush=True)


if __name__ == "__main__":
    main()
