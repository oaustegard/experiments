"""Re-embed the 61,686 posts of the Bluesky AI-list index as fp32 768-d EG2 q4 vectors.

Usage: python3 -I run/embed_docs.py

The published index (../embeddinggemma2-webgpu-tail/web/bsky) ships only 2-bit codes, and its
fp32 vectors stayed in that experiment's gitignored out/. Exact fp32 search is the ground truth
here, so the vectors are rebuilt: text from ../eg2-query-student/out/posts.jsonl
(run/hydrate_posts.py there), the document prompt "title: none | text: ", 128 tokens, the
model_q4.onnx graph on onnxruntime's CPU provider (eg2-query-student's Teacher, the same graph
build_bsky_index.py used). Deleted posts (text null) get a zero row and are listed in
out/missing.json. Checkpoints out/emb/chunk_NNNNN.npy every 2,048 posts; a rerun resumes.
Writes out/docs.npy (n x 768 float32, unit rows).
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "eg2-query-student" / "run"))
from common import Teacher  # noqa: E402

OUT = HERE / "out"
EMB = OUT / "emb"
POSTS = HERE.parent / "eg2-query-student" / "out" / "posts.jsonl"
DOC = "title: none | text: "
CHUNK = 2048


def main() -> int:
    rows = [json.loads(l) for l in POSTS.open()]
    assert [r["i"] for r in rows] == list(range(len(rows)))
    missing = [r["i"] for r in rows if r["text"] is None]
    EMB.mkdir(parents=True, exist_ok=True)
    (OUT / "missing.json").write_text(json.dumps(missing))
    print(f"{len(rows)} posts, {len(missing)} missing", flush=True)
    t = Teacher(threads=4)
    t0 = time.time()
    for c0 in range(0, len(rows), CHUNK):
        path = EMB / f"chunk_{c0 // CHUNK:05d}.npy"
        if path.exists():
            continue
        part = rows[c0:c0 + CHUNK]
        live = [j for j, r in enumerate(part) if r["text"] is not None]
        out = np.zeros((len(part), 768), np.float32)
        out[live] = t.embed([part[j]["text"] for j in live], prompt=DOC, max_tokens=128, batch=32)
        np.save(path, out)
        print(f"embedded {c0 + len(part)}/{len(rows)} in {time.time() - t0:.0f} s", flush=True)
    X = np.concatenate([np.load(p) for p in sorted(glob.glob(str(EMB / "chunk_*.npy")))])
    assert len(X) == len(rows)
    np.save(OUT / "docs.npy", X)
    return 0


if __name__ == "__main__":
    sys.exit(main())
