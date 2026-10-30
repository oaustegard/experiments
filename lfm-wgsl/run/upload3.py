"""Split model3/web/lfm3-gq4.bin into 64 MB parts and upload them to R2 (austegard-media, public, CORS for
austegard.com) under lfm3/gq4-v1/, using the hub's embstore helper. Records the part URLs in web/manifest3.json."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/user/claude-workspace/scripts")
import embstore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model3" / "web" / "lfm3-gq4.bin"
PREFIX = "lfm3/gq4-v1"
PART = 64 << 20
man_p = ROOT / "web" / "manifest3.json"
man = json.loads(man_p.read_text())
parts = []
with open(SRC, "rb") as f:
    i = 0
    while True:
        data = f.read(PART)
        if not data:
            break
        key = f"{PREFIX}/lfm3-gq4.bin.{i:03d}"
        if embstore._remote_size(key) != len(data):
            with tempfile.NamedTemporaryFile(delete=False) as t:
                t.write(data)
            embstore._put(key, Path(t.name), "application/octet-stream")
            Path(t.name).unlink()
        parts.append({"url": f"{embstore.PUBLIC}/{key}", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(key, len(data), flush=True)
        i += 1
man["parts"] = parts
man_p.write_text(json.dumps(man, separators=(",", ":")))
print(len(parts), "parts")
