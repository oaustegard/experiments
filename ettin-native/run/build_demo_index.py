"""Build a Bluesky search index for a web/models/<pkg>/ package, in the EG2 index's own format, for web/demo.html.

Usage: .venv/bin/python -I run/build_demo_index.py PKG [PKG ...]      e.g. e17c-t4-w8L63 leafirc-t4-w8L63

Rows are embstore-free and aligned with embeddinggemma2-webgpu-tail/web/bsky (61,686 posts; the
same posts.json and authors serve every index). Each post's hydrated text (eg2-query-student
out/posts.jsonl) is embedded with the package's own model.onnx (CPU onnxruntime, doc prompt,
128 tokens), then quantized exactly as run/build_bsky_index.py does there: remex
Quantizer(768, bits=2, seed=42, mean=corpus mean), packed codes (192 B/doc) and (scale, bias)
floats. The RHT rotation depends only on the seed and dim, so assets/params.json is shared.
Writes web/demo/index/PKG/{codes.u8.txt, aux.f32.txt, index.json}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import remex
from remex.packing import pack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_onnx import Enc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
POSTS = EXP / "eg2-query-student" / "out" / "posts.jsonl"
EG2 = EXP / "embeddinggemma2-webgpu-tail" / "web" / "bsky"


def main():
    rows = [json.loads(l) for l in POSTS.open()]
    eg2 = json.loads((EG2 / "index.json").read_text())
    plist = json.loads((EG2 / "posts.json").read_text())
    assert len(rows) == eg2["n"] == len(plist) and all(r["i"] == i for i, r in enumerate(rows))
    texts = [r["text"] or "" for r in rows]
    for pkg in sys.argv[1:]:
        d = ROOT / "web" / "models" / pkg
        enc = Enc(d)
        t0 = time.time()
        X = enc.embed(texts, enc.dp)
        mu = X.mean(0).astype(np.float32)
        qz = remex.Quantizer(768, bits=2, seed=42, mean=mu)
        cv = qz.encode(X)
        packed = pack(cv.indices.astype(np.uint8), 2).reshape(len(X), 192)
        aux = np.stack([qz._score_norms(cv), qz._row_bias(cv)], 1).astype("<f4")
        out = ROOT / "web" / "demo" / "index" / pkg
        out.mkdir(parents=True, exist_ok=True)
        (out / "codes.u8.txt").write_bytes(np.ascontiguousarray(packed).tobytes())
        (out / "aux.f32.txt").write_bytes(aux.tobytes())
        (out / "index.json").write_text(json.dumps({
            "n": len(X), "d": 768, "bits": 2, "seed": 42, "rotation": "rht", "remex_version": remex.__version__,
            "model": f"ettin-native web/models/{pkg}/model.onnx", "doc_prompt": enc.dp, "query_prompt": enc.qp,
            "max_tokens": 128, "rows": "embeddinggemma2-webgpu-tail/web/bsky/posts.json",
            "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "embed_s": round(time.time() - t0),
            "mu": [float(v) for v in mu], "files": {"codes": "codes.u8.txt", "aux": "aux.f32.txt"},
        }, separators=(",", ":")))
        print(pkg, len(X), f"{time.time() - t0:.0f} s", packed.nbytes, flush=True)


if __name__ == "__main__":
    main()
