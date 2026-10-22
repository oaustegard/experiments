"""Bundle what web/demo/ needs to run offline: post texts and the EG2 4-layer cut's files.

Usage: .venv/bin/python -I run/build_demo_data.py

1. web/demo/data/corpus.json: authors [[handle, did]], posts [[author, rkey]] (the EG2 index's
   row order) and texts, so result cards render without asking Bluesky. Texts are the hydrated
   posts of eg2-query-student (out/posts.jsonl), the same text the Ettin and leaf-ir indexes embed.
2. web/demo/eg2cut/: model_t3w.onnx (+ _data), tokenizer.json, meta.json from release
   eg2-layer-onnx-20261021 (eg2cut-k4.zip, sha256 checked against its SHA256SUMS.json), and
   eg2-wgsl-engine/web/manifest.json, whose byte offsets index model_t3w.onnx_data.
Both are gitignored (posts are other people's words; the model is 25 MB): rebuild with this script.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
EG2 = EXP / "embeddinggemma2-webgpu-tail" / "web" / "bsky"
R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/models/eg2-layer-student/eg2-layer-onnx-20261021/"
UA = {"User-Agent": "ettin-native/1"}  # r2.dev refuses urllib's default agent


def main():
    out = ROOT / "web" / "demo"
    (out / "data").mkdir(parents=True, exist_ok=True)
    idx = json.loads((EG2 / "index.json").read_text())
    plist = json.loads((EG2 / "posts.json").read_text())
    rows = [json.loads(l) for l in (EXP / "eg2-query-student" / "out" / "posts.jsonl").open()]
    assert len(rows) == len(plist) == idx["n"]
    for r, (a, rkey) in zip(rows, plist):  # same post in the same row
        assert r["uri"] == f"at://{idx['authors'][a]['did']}/app.bsky.feed.post/{rkey}", r["uri"]
    corpus = {"authors": [[x["handle"], x["did"]] for x in idx["authors"]], "posts": plist,
              "texts": [r["text"] or "" for r in rows]}
    (out / "data" / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, separators=(",", ":")))
    print("corpus.json", (out / "data" / "corpus.json").stat().st_size, flush=True)

    cut = out / "eg2cut"
    cut.mkdir(exist_ok=True)
    if not (cut / "model_t3w.onnx_data").exists():
        sums = json.loads(urllib.request.urlopen(urllib.request.Request(R2 + "SHA256SUMS.json", headers=UA)).read())
        blob = urllib.request.urlopen(urllib.request.Request(R2 + "eg2cut-k4.zip", headers=UA)).read()
        assert hashlib.sha256(blob).hexdigest() == sums["eg2cut-k4.zip"]["sha256"], "eg2cut-k4.zip sha256 mismatch"
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for f in ("model_t3w.onnx", "model_t3w.onnx_data", "tokenizer.json", "meta.json"):
                (cut / f).write_bytes(z.read(f"eg2cut-k4/{f}"))
    (cut / "manifest.json").write_bytes((EXP / "eg2-wgsl-engine" / "web" / "manifest.json").read_bytes())
    print("eg2cut", {f.name: f.stat().st_size for f in cut.iterdir()})


if __name__ == "__main__":
    sys.exit(main())
