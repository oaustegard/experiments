"""Fetch everything a fresh checkout needs to train and score students, no credentials.

Usage: python3 -I run/fetch_data.py

Into ../eg2-query-student/out/ (gitignored), where run/model.py and #102's scripts look:
  hf/        onnx-community/embeddinggemma-2-ONNX model_q4.onnx + data, tokenizer (the teacher)
  eg2pt/     google/embeddinggemma-2 config + model.safetensors (norms and config)
  targets/   embstore bsky-ai-list/embeddinggemma-2-q4/query-targets-v1, sha256-checked,
             from the public R2 bucket; queries/ links to it
  posts.jsonl  post text re-fetched from Bluesky (eg2-query-student/run/hydrate_posts.py)
Every step skips what is already there, so a rerun resumes.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
PREV = ROOT.parent / "eg2-query-student"
OUT = PREV / "out"
R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/embeddings/bsky-ai-list/embeddinggemma-2-q4/query-targets-v1"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def get(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-experiments"})  # r2.dev 403s urllib's default UA
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dst)


def main():
    (OUT / "hf").mkdir(parents=True, exist_ok=True)
    dl = OUT / "hf_dl"
    for f in ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json", "tokenizer_config.json", "config.json"]:
        dst = OUT / "hf" / Path(f).name
        if not dst.exists():
            shutil.copy(hf_hub_download("onnx-community/embeddinggemma-2-ONNX", f, local_dir=str(dl)), dst)
    for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        if not (OUT / "eg2pt" / f).exists():
            hf_hub_download("google/embeddinggemma-2", f, local_dir=str(OUT / "eg2pt"))
    print("models ok", flush=True)
    tg = OUT / "targets"
    tg.mkdir(exist_ok=True)
    if not (tg / "index.json").exists():
        get(R2 + "/index.json", tg / "index.json")
    for f in json.loads((tg / "index.json").read_text())["files"]:
        p = tg / f["name"]
        if p.exists() and p.stat().st_size == f["bytes"]:
            continue
        get(f"{R2}/{f['name']}", p)
        if sha256(p) != f["sha256"]:
            sys.exit(f"sha256 mismatch: {f['name']}")
        print("got", f["name"], flush=True)
    q = OUT / "queries"
    if not q.exists():
        q.symlink_to("targets")
    print("targets ok", flush=True)
    if not (OUT / "posts.jsonl").exists() or sum(1 for _ in (OUT / "posts.jsonl").open()) < 61686:
        subprocess.run([sys.executable, "-I", str(PREV / "run" / "hydrate_posts.py")], check=True)
    print("posts ok", flush=True)


if __name__ == "__main__":
    main()
