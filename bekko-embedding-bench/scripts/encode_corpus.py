"""Encode a chunk set with a bekko variant, checkpointing as it goes.

METHODS.md: CCotw silently reaps long-running background jobs on idle, so a
multi-minute embedding run must checkpoint mid-run rather than hold everything
in memory until the end. Vectors go to a float32 memmap written incrementally;
a rerun resumes from the first unwritten row.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--mode", required=True, choices=["ast", "flat"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    chunks = json.load(open(HERE / f"chunks_{a.mode}.json"))
    texts = [c["text"] for c in chunks]
    n = len(texts)

    out = HERE / f"vecs_{a.mode}_{a.variant}.f32"
    done_p = HERE / f"vecs_{a.mode}_{a.variant}.done"
    dim = 384
    mm = np.memmap(out, dtype=np.float32, mode="r+" if out.exists() else "w+", shape=(n, dim))
    start = int(done_p.read_text()) if done_p.exists() else 0
    if start >= n:
        print(f"{a.mode}/{a.variant}: already complete ({n})", flush=True)
        return

    enc = BekkoEncoder(a.variant, threads=a.threads)
    t0 = time.time()
    CK = 512
    for s in range(start, n, CK):
        block = texts[s : s + CK]
        mm[s : s + len(block)] = enc.encode(block, batch_size=a.batch)
        mm.flush()
        done_p.write_text(str(s + len(block)))
        el = time.time() - t0
        rate = (s + len(block) - start) / max(el, 1e-9)
        print(
            f"{a.mode}/{a.variant}: {s + len(block)}/{n} "
            f"{rate:.1f} ch/s eta {(n - s - len(block)) / max(rate, 1e-9) / 60:.1f}m",
            flush=True,
        )
    print(f"{a.mode}/{a.variant}: DONE {n} in {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
