"""Publish the fine-tuned 8-bit weights for the chat page, next to A8p_q8 and base_q8.

    python3 upload_r2.py export/skygest_e8_q8_page skygest_q8

Writes models/monad-chat/v1/<name>/meta.json and model.packed in the public bucket austegard-media
(claude-workspace scripts/embstore.py _put, CF_API_TOKEN). The graph and tokenizer already there are
shared: this refuses to upload unless the export's graph_sha256 equals the published model.onnx's.
Then reads both files back from the public URL and checks size and sha256.
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402,F401

sys.path.insert(0, "/home/user/claude-workspace/scripts")
import embstore  # noqa: E402

PREFIX = "models/monad-chat/v1"


def get(key):
    req = urllib.request.Request(f"{embstore.PUBLIC}/{key}", headers={"User-Agent": "muninn-experiments"})
    return urllib.request.urlopen(req, timeout=300).read()


def main():
    src, name = HERE / sys.argv[1], sys.argv[2]
    meta = json.loads((src / "meta.json").read_text())
    live = hashlib.sha256(get(f"{PREFIX}/model.onnx")).hexdigest()
    assert meta["graph_sha256"] == live, (meta["graph_sha256"], live)
    packed = src / "model.packed"
    assert packed.stat().st_size == meta["packed_bytes"]
    embstore._put(f"{PREFIX}/{name}/meta.json", src / "meta.json", "application/json")
    embstore._put(f"{PREFIX}/{name}/model.packed", packed, "application/octet-stream")
    for f, local in (("meta.json", src / "meta.json"), ("model.packed", packed)):
        b = get(f"{PREFIX}/{name}/{f}")
        ok = hashlib.sha256(b).hexdigest() == hashlib.sha256(local.read_bytes()).hexdigest()
        print(json.dumps({"key": f"{PREFIX}/{name}/{f}", "bytes": len(b), "sha256_match": ok}))
        assert ok


if __name__ == "__main__":
    main()
