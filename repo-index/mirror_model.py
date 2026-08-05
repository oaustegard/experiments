#!/usr/bin/env python3
"""Publish the pinned encoder to this repo's releases, so CI does not depend on
Hugging Face availability or rate limits.

bekko-embedding-v1-a8m is MIT-licensed, so redistribution is permitted.

    gh release create repo-index-model-v1 --notes "pinned encoder for repo-index"
    python3 repo-index/mirror_model.py | xargs -n2 gh release upload repo-index-model-v1
"""
import hashlib
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".cache/repo-index")
for name in ("model.onnx", "tokenizer.json", "config.json"):
    p = SRC / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run ask.py once to populate the cache")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"{p}  # sha256 {h}")
