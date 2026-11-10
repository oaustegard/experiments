"""Write the 256-dim head for a published Ettin-17m package: fold256.f32 and fold256.json.

Usage: python3 -I run/site_fold.py MODEL_DIR
       (MODEL_DIR holds model.onnx and manifest.json, e.g. SITE/bsky/embedders/models/ettin)

The package's model ends in e = p W: p the pooled 256 numbers, W the 256 x 768 head, stored
as int8 codes with one scale per output column at the byte ranges manifest.json gives. With
W = A S B^T (SVD) and R remex's rotation for d = 256, seed 42, the head G = A S R^T gives
p G = e B R^T. C = B R^T has orthonormal columns, so the unit 256-dim output is the unit
768-dim output times C: every inner product is unchanged, and the coordinates are already
the ones remex would quantize.

fold256.f32 is G transposed, float32 little-endian, [N = 256][K = 256]: row n holds the
weights of output n over the pooled state, the layout the WGSL engine's f32 matmul reads.
The engine runs it in place of the 768-wide head (engine.js foldHead). model.onnx is not
changed, so pages that use 768 dims keep working from the same file.

The script checks the head it wrote against model.onnx run by onnxruntime on a few texts
when MODEL_DIR also has tokenizer.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import remex

SEED = 42


def head_weight(model_dir):
    m = json.loads((model_dir / "manifest.json").read_text())
    d = m["dense"]
    assert m["dim"] == 768 and d["K"] == 256 and d["N"] == 768 and "q" in d and "b" not in d, "expected a bias-free int8 256 -> 768 head"
    raw = (model_dir / "model.onnx").read_bytes()
    q = np.frombuffer(raw, np.int8, d["K"] * d["N"], d["q"]["offset"]).reshape(d["K"], d["N"])
    s = np.frombuffer(raw, "<f4", d["N"], d["s"]["offset"])
    return q.astype(np.float64) * s.astype(np.float64), hashlib.sha256(raw).hexdigest()


def write(model_dir, check=True):
    """Write fold256.f32 and fold256.json into model_dir; returns the metadata."""
    model_dir = Path(model_dir).resolve()
    W, sha = head_weight(model_dir)
    A, S, Bt = np.linalg.svd(W, full_matrices=False)
    R = remex.Quantizer(256, bits=4, seed=SEED).R.astype(np.float64)
    assert np.abs(R @ R.T - np.eye(256)).max() < 1e-6
    G = (A * S) @ R.T                      # [K, N']: z = p G
    C = Bt.T @ R.T                         # [768, 256]: y = x C
    assert np.abs(C.T @ C - np.eye(256)).max() < 1e-9
    assert np.abs(W @ C - G).max() < 1e-9
    head = np.ascontiguousarray(G.T, "<f4")  # [N, K]
    (model_dir / "fold256.f32").write_bytes(head.tobytes())
    meta = {
        "dim": 256, "from_dim": 768, "file": "fold256.f32", "layout": "float32 little-endian [N=256][K=256], y_n = sum_k pooled_k w[n][k]",
        "basis": "an orthonormal basis of the 256 directions the dequantized 256 x 768 head's outputs span (SVD), then remex's rotation",
        "rotation": {"library": "remex", "version": remex.__version__, "d": 256, "seed": SEED, "kind": remex.Quantizer(256, bits=4, seed=SEED).rotation},
        "head_singular_values": [float(S[0]), float(S[-1])],
        "model_sha256": sha, "sha256": hashlib.sha256(head.tobytes()).hexdigest(),
        "built_by": "oaustegard/experiments ettin-span256-remex/run/site_fold.py",
    }
    (model_dir / "fold256.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))

    if check and (model_dir / "tokenizer.json").exists():
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # the graph's own arithmetic
        sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        texts = ["claude coding agent costs", "a soap film simulated on the GPU", "how do I fold a projection head",
                 "Represent this sentence for searching relevant passages: vector quantization", "x"]
        X = []
        for t in texts:
            ids = np.array([tok.encode(t).ids], np.int64)
            e = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0].astype(np.float64)
            X.append(e / np.linalg.norm(e))
        X = np.array(X)
        Y = X @ C
        gram = float(np.abs(Y @ Y.T - X @ X.T).max())
        norm = float(np.abs(np.linalg.norm(Y, axis=1) - 1).max())
        print(f"check on {len(texts)} texts: max |gram256 - gram768| {gram:.1e}, max |norm - 1| {norm:.1e}")
        assert gram < 1e-6 and norm < 1e-6
    return meta


if __name__ == "__main__":
    write(sys.argv[1])
