"""CPU reference for the real-model demo run: EG2 model_q4.onnx on onnxruntime's CPU provider.

Usage: python3 -I run/ref_real.py

Embeds the model card's text-search example (one query, four documents) with the same
tokenizer.json and graph the demo loads, and writes out/ref_real/ref.json. run/run_demo.py --real
compares the browser's WebGPU embeddings against it. Needs onnxruntime and tokenizers; downloads
the model files into out/hf/ on first use (175 MB).
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
HF = "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/"
FILES = ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json", "tokenizer_config.json"]
CACHE = ROOT / "out" / "hf"
OUT = ROOT / "out" / "ref_real"

# The model card's example and the scores it prints (rounded to 3 places).
QUERY = "task: search result | query: Which planet is known as the Red Planet?"
DOCS = [
    "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
    "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "title: none | text: Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "title: none | text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
CARD_SCORES = [0.684, 0.854, 0.752, 0.783]


def fetch(name: str) -> Path:
    dst = CACHE / Path(name).name
    if not dst.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(HF + name, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dst)
    return dst


def main() -> int:
    paths = {Path(n).name: fetch(n) for n in FILES}
    tok = Tokenizer.from_file(str(paths["tokenizer.json"]))
    sess = ort.InferenceSession(str(paths["model_q4.onnx"]), providers=["CPUExecutionProvider"])

    texts = [QUERY] + DOCS
    enc = [tok.encode(t).ids for t in texts]
    S = max(map(len, enc))
    ids = np.zeros((len(texts), S), np.int64)
    attn = np.zeros((len(texts), S), np.int64)
    for b, row in enumerate(enc):
        ids[b, : len(row)] = row
        attn[b, : len(row)] = 1
    empty = np.zeros((0, 512), np.float32)
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, sess.run(None, {
        "input_ids": ids, "attention_mask": attn,
        "image_features": empty, "video_features": empty, "audio_features": empty,
    })))
    emb = out["sentence_embedding"].astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    scores = (emb[1:] @ emb[0]).tolist()

    OUT.mkdir(parents=True, exist_ok=True)
    ref = {
        "texts": texts, "token_ids": enc, "card_scores": CARD_SCORES,
        "cpu_scores": scores, "embeddings": emb.tolist(),
    }
    (OUT / "ref.json").write_text(json.dumps(ref) + "\n")
    print(json.dumps({"cpu_scores": [round(s, 4) for s in scores], "card_scores": CARD_SCORES}))
    ok = all(abs(a - b) <= 6e-4 for a, b in zip(scores, CARD_SCORES))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
