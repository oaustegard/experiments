"""Build bridge/spec.bin for job.js from saved runs.

    python3 bridge/make_job.py RUN.json[:n1,n2,...] ... -o bridge/spec.bin
Each RUN.json gives shaders at the listed prefix lengths (default: full). Residual runs
include their residual at full length.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import codec  # noqa: E402

SIZES = [[768, 512], [1920, 1280], [3840, 2560]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    spec = []
    for item in a.runs:
        path, _, ns = item.partition(":")
        run, res = codec.load_run(path)
        N = int(run["P"].shape[0])
        for n in ([int(x) for x in ns.split(",")] if ns else [N]):
            r = res if n == N else None
            wgsl = codec.rate(codec.emit_wgsl(run, n=n, residual=r))["wgsl"]
            sid = f"{Path(path).stem}_n{n}" + ("_res" if r is not None else "")
            spec.append({"id": sid, "wgsl": wgsl, "w": run["W"], "h": run["H"], "sizes": SIZES, "reps": a.reps,
                         "run": path, "n": n})
    Path(a.out).write_text(json.dumps(spec))
    print(len(spec), "shaders:", [s["id"] for s in spec])


if __name__ == "__main__":
    main()
