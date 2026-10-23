"""Embed the RAM++ tag list with MobileCLIP-S0's text tower, offline.

The page runs only the vision tower; these vectors are its label bank. Uses the same
Xenova/mobileclip_s0 text_model.onnx (fp32) that transformers.js would load, padded to 77
tokens with the pad token, as CLIPTokenizer does. Each tag is the mean of a few prompt
templates (CLIP's prompt ensembling), renormalized.

Writes web/tags.f16.bin (N x 512 float16, row-major, unit norm) and web/tags.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / ".cache"
TEMPLATES = ["a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}.", "{}"]
CTX = 77


def encode(tok, sess, texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        ids = np.zeros((len(texts[i:i + batch]), CTX), dtype=np.int64)  # pad token "!" is id 0
        for r, t in enumerate(texts[i:i + batch]):
            e = tok.encode(t).ids[:CTX]
            e[-1] = 49407 if len(e) == CTX else e[-1]
            ids[r, :len(e)] = e
        out.append(sess.run(None, {"input_ids": ids})[0])
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    tags = [t.strip() for t in (CACHE / "ram_tag_list.txt").read_text().splitlines() if t.strip()]
    seen, uniq = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower()); uniq.append(t)
    tok = Tokenizer.from_file(str(CACHE / "tokenizer.json"))
    sess = ort.InferenceSession(str(CACHE / "onnx/text_model.onnx"))
    acc = np.zeros((len(uniq), 512), dtype=np.float64)
    for tpl in TEMPLATES:
        acc += encode(tok, sess, [tpl.format(t) for t in uniq])
        print(tpl, file=sys.stderr)
    bank = (acc / np.linalg.norm(acc, axis=1, keepdims=True)).astype(np.float32)
    np.save(CACHE / "tags.f32.npy", bank)
    (HERE / "web/tags.f16.bin").write_bytes(bank.astype(np.float16).tobytes())
    (HERE / "web/tags.json").write_text(json.dumps({
        "source": "RAM++ tag list, github.com/xinyu1205/recognize-anything ram/data/ram_tag_list.txt",
        "model": "Xenova/mobileclip_s0 onnx/text_model.onnx (fp32)",
        "templates": TEMPLATES, "dim": 512, "dtype": "float16", "tags": uniq}))
    print(len(uniq), "tags", file=sys.stderr)


if __name__ == "__main__":
    main()
