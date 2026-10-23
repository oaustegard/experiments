"""Embed the RAM++ tag list with MobileCLIP-S0's text tower, offline.

The page runs only the vision tower; these vectors are its label bank. Uses the same
Xenova/mobileclip_s0 text_model.onnx (fp32) that transformers.js would load, padded to 77
tokens with the pad token, as CLIPTokenizer does. Each tag is the mean of a few prompt
templates (CLIP's prompt ensembling), renormalized.

Writes web/tags.f16.bin (N x 512 float16, row-major, unit norm) and web/tags.json.

`--vocab big` adds Open Images V7's 20,931 class display names (oidv7-class-descriptions.csv)
to the RAM++ list, deduplicated case-insensitively, and writes web/tags-big.i8.bin instead:
N x 512 int8 rows (symmetric, per-row scale max|x|/127) followed by N float32 scales.
Embeddings are cached per tag string (.cache/tag_emb_cache.json + .npy), so a rerun only embeds new tags.
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


def vocab(name):
    tags = [t.strip() for t in (CACHE / "ram_tag_list.txt").read_text().splitlines() if t.strip()]
    if name == "big":
        import csv
        with open(CACHE / "oidv7-class-descriptions.csv", newline="") as f:
            tags += [r["DisplayName"].strip() for r in csv.DictReader(f) if r["DisplayName"].strip()]
    return tags


def main(name):
    tags = vocab(name)
    seen, uniq = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower()); uniq.append(t)
    tok = Tokenizer.from_file(str(CACHE / "tokenizer.json"))
    sess = ort.InferenceSession(str(CACHE / "onnx/text_model.onnx"))
    keys_path, vecs_path = CACHE / "tag_emb_cache.json", CACHE / "tag_emb_cache.npy"
    cache = dict(zip(json.loads(keys_path.read_text()), np.load(vecs_path))) if keys_path.exists() else {}
    todo = [t for t in uniq if t not in cache]
    if todo:
        acc = np.zeros((len(todo), 512), dtype=np.float64)
        for tpl in TEMPLATES:
            acc += encode(tok, sess, [tpl.format(t) for t in todo])
            print(tpl, len(todo), file=sys.stderr)
        for t, v in zip(todo, acc / np.linalg.norm(acc, axis=1, keepdims=True)):
            cache[t] = v.astype(np.float32)
        keys = list(cache)  # not np.savez(**cache): a tag can be named "file"
        np.save(vecs_path, np.stack([cache[k] for k in keys])); keys_path.write_text(json.dumps(keys))
    bank = np.stack([cache[t] for t in uniq]).astype(np.float32)
    meta = {"model": "Xenova/mobileclip_s0 onnx/text_model.onnx (fp32)", "templates": TEMPLATES, "dim": 512, "tags": uniq}
    if name == "big":
        scale = np.abs(bank).max(axis=1) / 127
        q = np.round(bank / scale[:, None]).astype(np.int8)
        (HERE / "web/tags-big.i8.bin").write_bytes(q.tobytes() + scale.astype(np.float32).tobytes())
        n_ram = len({t.lower() for t in vocab("ram")})
        assert [t.lower() for t in uniq[:n_ram]] == list(dict.fromkeys(t.lower() for t in vocab("ram")))
        meta |= {"source": "RAM++ tag list + Open Images V7 class display names", "dtype": "int8+rowscale",
                 "n_general": n_ram}  # rows [0, n_general) are RAM++, the rest Open Images only
        (HERE / "web/tags-big.json").write_text(json.dumps(meta))
    else:
        (HERE / "web/tags.f16.bin").write_bytes(bank.astype(np.float16).tobytes())
        meta |= {"source": "RAM++ tag list, github.com/xinyu1205/recognize-anything ram/data/ram_tag_list.txt", "dtype": "float16"}
        (HERE / "web/tags.json").write_text(json.dumps(meta))
    print(len(uniq), "tags", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--vocab") + 1] if "--vocab" in sys.argv else "ram")
