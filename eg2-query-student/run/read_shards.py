"""Read the shard files web/index.html saves and turn them into training files.

Usage: python3 -I run/read_shards.py DIR   (DIR holds eg2_targets_NN.bin)

Writes out/queries/gpu_more.jsonl + gpu_more.eg2.npy (the synth_more queries) and
out/queries/gpu_posts.jsonl + gpu_posts.eg2.npy (post snippets), in the same layout the
CPU teacher writes, so a1_linear.py and a3_distill.py take them as --train names. A post
row is kept only when the page embedded it and its snippet length matches the snippet built
here from out/posts.jsonl (a post edited or deleted since hydration is dropped).
Also checks every shard header's parity block and prints the per-shard counts.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_web import snippet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = 768


def read(path):
    b = path.read_bytes()
    assert b[:4] == b"EGQT", path
    H = struct.unpack("<I", b[4:8])[0]
    h = json.loads(b[8:8 + H])
    n, base = h["n"], 8 + H
    codes = np.frombuffer(b, np.int8, n * D, base).reshape(n, D).astype(np.float32)
    off = base + n * D
    scale = np.frombuffer(b, "<f4", n, off)
    off += 4 * n
    status = np.frombuffer(b, np.uint8, n, off)
    off += n + (off + n) % 2
    lens = np.frombuffer(b, "<u2", n, off)
    v = codes * scale[:, None]
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return h, np.where(nrm > 0, v / np.maximum(nrm, 1e-12), 0), status, lens


def main():
    src = Path(sys.argv[1])
    items = json.loads((ROOT / "web" / "items.json").read_text())["items"]
    texts = {json.loads(l)["i"]: json.loads(l)["text"] for l in (ROOT / "out" / "posts.jsonl").open()}
    q_rows, q_vecs, p_rows, p_vecs = [], [], [], []
    drift = 0
    for f in sorted(src.glob("eg2_targets_*.bin")):
        h, V, st, ln = read(f)
        assert h["parity"]["min_cos"] >= 0.995, (f, h["parity"])
        for j in range(h["n"]):
            it = items[h["start"] + j]
            if not st[j]:
                continue
            if isinstance(it, str):
                q_rows.append({"id": f"more{h['start'] + j}", "q": it, "kind": "synth_more"})
                q_vecs.append(V[j])
            else:
                s = snippet(texts[it] or "")
                if len(s) != ln[j]:
                    drift += 1
                    continue
                p_rows.append({"id": f"snip{it}", "q": s, "kind": "post_snippet", "post": it})
                p_vecs.append(V[j])
        print(f.name, {k: h[k] for k in ("shard", "n", "embedded", "missing_posts", "embed_ms")},
              "parity", round(h["parity"]["min_cos"], 6))
    Q = ROOT / "out" / "queries"
    for name, rows, vecs in [("gpu_more", q_rows, q_vecs), ("gpu_posts", p_rows, p_vecs)]:
        with (Q / f"{name}.jsonl").open("w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        np.save(Q / f"{name}.eg2.npy", np.array(vecs, np.float32).reshape(-1, D))
    print(json.dumps({"queries": len(q_rows), "snippets": len(p_rows), "snippet_drift_dropped": drift}))


if __name__ == "__main__":
    main()
