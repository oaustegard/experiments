#!/usr/bin/env python3
"""Download PaddlePaddle/PP-OCRv6_{tiny,small}_{det,rec}_onnx into models/<size>_<kind>/.

    python3 -I run/fetch_models.py                 # all four
    python3 -I run/fetch_models.py tiny_rec small_det

Each model directory gets inference.onnx (fp32 ONNX) and inference.yml (pre/post-processing config; the
recognizer's character dictionary lives under PostProcess.character_dict). Files already present with the
expected size are kept. Apache-2.0 upstream (see the model cards).
"""
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
URL = "https://huggingface.co/PaddlePaddle/PP-OCRv6_{size}_{kind}_onnx/resolve/main/{file}"
FILES = ("inference.onnx", "inference.yml")
# byte sizes from the Hub tree API (2026-11-24); a mismatch means a changed upstream file or a cut transfer
SIZES = {
    "tiny_det": {"inference.onnx": 1780590, "inference.yml": 883},
    "tiny_rec": {"inference.onnx": 4462639, "inference.yml": 55571},
    "small_det": {"inference.onnx": 9880512, "inference.yml": 885},
    "small_rec": {"inference.onnx": 21159378, "inference.yml": 150579},
}


def fetch(url, dest, expect, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ppocr-pack/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != expect:
                raise OSError(f"{len(data)} bytes, expected {expect}")
            dest.write_bytes(data)
            return hashlib.sha256(data).hexdigest()
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"retry {url}: {e}", file=sys.stderr)
            time.sleep(2 ** k)


def main(argv):
    names = argv or list(SIZES)
    for name in names:
        size, kind = name.split("_")
        d = MODELS / name
        d.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            dest = d / f
            if dest.exists() and dest.stat().st_size == SIZES[name][f]:
                print(f"{name}/{f}: present ({dest.stat().st_size} bytes)")
                continue
            sha = fetch(URL.format(size=size, kind=kind, file=f), dest, SIZES[name][f])
            print(f"{name}/{f}: downloaded {dest.stat().st_size} bytes sha256 {sha[:16]}")


if __name__ == "__main__":
    main(sys.argv[1:])
