"""EG2 query embeddings (the teacher targets) for one or more query files.

Usage: python3 -I run/embed_teacher.py out/queries/real_test.jsonl [more.jsonl ...]

For each X.jsonl writes X.eg2.npy (n x 768 float32, unit, query prompt) in 2,048-row
checkpoint chunks under X.eg2.parts/; a rerun resumes.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Teacher  # noqa: E402

CHUNK = 2048


def main():
    T = Teacher()
    for p in map(Path, sys.argv[1:]):
        qs = [json.loads(l)["q"] for l in p.open()]
        parts = p.with_suffix(".eg2.parts")
        parts.mkdir(exist_ok=True)
        t0 = time.time()
        for s in range(0, len(qs), CHUNK):
            f = parts / f"{s // CHUNK:05d}.npy"
            if not f.exists():
                np.save(f, T.embed(qs[s:s + CHUNK]))
                print(f"{p.name} {s + CHUNK}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
        E = np.concatenate([np.load(f) for f in sorted(parts.glob("*.npy"))])
        assert len(E) == len(qs)
        np.save(p.with_suffix(".eg2.npy"), E)


if __name__ == "__main__":
    main()
