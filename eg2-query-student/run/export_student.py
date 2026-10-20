"""Export a trained A3 student to ONNX, quantize weights to int8, and score what ships.

Usage: python3 -I run/export_student.py out/a3/<tag>/epoch<k> [--prompt "..."]

Writes <ckpt>/onnx/model.onnx (fp32) and model_int8.onnx (onnxruntime quantize_dynamic,
per-channel int8 weights), then embeds both test sets through the int8 graph with the
checkpoint's own tokenizer and scores top-10 overlap against EG2 exactly as the arms do.
Appends to results/export.json: bytes of each file, of the tokenizer, and the overlaps.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_linear import kinds, score  # noqa: E402
from common import Index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "out" / "queries"


class Wrap(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, input_ids, attention_mask):
        return self.st({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--prompt", default=None, help="query prompt; default: the checkpoint's own")
    a = ap.parse_args()
    ck = Path(a.ckpt).resolve()
    st = SentenceTransformer(str(ck), device="cpu").eval()
    prompt = a.prompt if a.prompt is not None else (st.prompts or {}).get("query", "")
    dst = ck / "onnx"
    dst.mkdir(exist_ok=True)
    tok = st.tokenizer
    ex = tok(["a short example query"], return_tensors="pt")
    torch.onnx.export(Wrap(st), (ex["input_ids"], ex["attention_mask"]), str(dst / "model.onnx"),
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "embedding": {0: "b"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(dst / "model.onnx"), str(dst / "model_int8.onnx"),
                     weight_type=QuantType.QInt8, per_channel=True)
    sess = ort.InferenceSession(str(dst / "model_int8.onnx"), providers=["CPUExecutionProvider"])
    ix = Index()
    tok_files = [f for f in ck.iterdir() if f.name in ("tokenizer.json", "vocab.txt")]
    r = {"ckpt": str(ck.relative_to(ROOT)), "prompt": prompt,
         "fp32_bytes": (dst / "model.onnx").stat().st_size,
         "int8_bytes": (dst / "model_int8.onnx").stat().st_size,
         "int8_gz_bytes": len(gzip.compress((dst / "model_int8.onnx").read_bytes())),
         "tokenizer_gz_bytes": {f.name: len(gzip.compress(f.read_bytes())) for f in tok_files}}
    for t in ["real_test", "synth_test"]:
        qs = [prompt + json.loads(l)["q"] for l in (Q / f"{t}.jsonl").open()]
        out = []
        for s in range(0, len(qs), 128):
            enc = tok(qs[s:s + 128], padding=True, truncation=True, max_length=128, return_tensors="np")
            out.append(sess.run(["embedding"], {"input_ids": enc["input_ids"].astype(np.int64),
                                                "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
        P = np.concatenate(out).astype(np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        r[t] = score(ix, P, np.load(Q / f"{t}.eg2.npy"), kinds(t))
    print(json.dumps({k: (v["overlap@10"] if isinstance(v, dict) and "overlap@10" in v else v)
                      for k, v in r.items()}))
    rp = ROOT / "results" / "export.json"
    allr = json.loads(rp.read_text()) if rp.exists() else {}
    allr[r["ckpt"]] = r
    rp.write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
