"""Publish the chat page's model files to the public R2 bucket austegard-media.

    python3 upload.py            # uploads under models/monad-chat/v1/

Layout (the page reads it from the bucket's r2.dev URL; CORS allows https://austegard.com):
    model.onnx                 graph shared by both weight sets (same graph_sha256)
    tokenizer.json
    <name>/meta.json           sizes, sha256 and the bf16 regions the page unpacks
    <name>/model.packed        weights, fp32 regions stored as bf16 where exact (see export.py)
for <name> in A8p_q8 (loop-edited, 8-bit) and base_q8 (original Monad, 8-bit).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/claude-workspace/scripts")
import embstore  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

EXP = experiment("monad-weight-edit") / "export"
PREFIX = "models/monad-chat/v1"
NAMES = ("A8p_q8", "base_q8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    g = {sha(EXP / n / "model.onnx") for n in NAMES}
    assert len(g) == 1, "graphs differ"
    for n in NAMES:
        m = json.loads((EXP / n / "meta.json").read_text())
        assert (EXP / n / "model.packed").stat().st_size == m["packed_bytes"]
    embstore._put(f"{PREFIX}/model.onnx", EXP / NAMES[0] / "model.onnx", "application/octet-stream")
    embstore._put(f"{PREFIX}/tokenizer.json", EXP / NAMES[0] / "tokenizer.json", "application/json")
    for n in NAMES:
        embstore._put(f"{PREFIX}/{n}/meta.json", EXP / n / "meta.json", "application/json")
        embstore._put(f"{PREFIX}/{n}/model.packed", EXP / n / "model.packed", "application/octet-stream")
        print("uploaded", n, flush=True)


if __name__ == "__main__":
    main()
