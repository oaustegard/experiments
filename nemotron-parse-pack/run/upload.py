"""Upload a pack directory to the public R2 bucket under nemotron-parse-pack/<name>/.

    python3 run/upload.py web/models/v1 v1
    python3 run/upload.py web/models/v35 v35 web/models/LICENSE35.txt     # a licence file other than 2.0's

Shards go up as application/octet-stream without Content-Encoding (the engine inflates them), the
manifest and tokenizer as JSON, and the model licence beside them. Uses whisper-pack/run/shard.py's put(),
which reads CF_ACCOUNT_ID and CF_API_TOKEN from the environment and passes the token in a header file
(shard.py parses its own arguments on import, so put() is copied here).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BUCKET = "austegard-media"
PUBLIC = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev"


def put(path, key, ctype):
    url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}/r2/buckets/{BUCKET}/objects/{key}"
    with tempfile.NamedTemporaryFile("w", delete=False) as h:
        os.chmod(h.name, 0o600)
        h.write(f"Authorization: Bearer {os.environ['CF_API_TOKEN']}\n")
    try:
        r = subprocess.run(["curl", "-sS", "-X", "PUT", "-H", f"@{h.name}", "-H", f"Content-Type: {ctype}",
                            "--data-binary", f"@{path}", url], capture_output=True, text=True)
    finally:
        os.unlink(h.name)
    if r.returncode != 0 or '"success":true' not in r.stdout.replace(" ", ""):
        sys.exit(f"PUT {key} failed: {r.stdout[:300]} {r.stderr[:300]}")


src, name = Path(sys.argv[1]), sys.argv[2]
prefix = f"nemotron-parse-pack/{name}/"
lic = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / "web" / "models" / "LICENSE.txt"
files = sorted(src.glob("weights-*.bin.gz")) + [src / "tokenizer.json", lic, src / "manifest.json"]
for f in files:
    ctype = "application/json" if f.suffix == ".json" else "text/plain; charset=utf-8" if f.suffix == ".txt" else "application/octet-stream"
    put(str(f), prefix + f.name, ctype)
    print("put", prefix + f.name, f.stat().st_size, flush=True)
print("public:", PUBLIC + "/" + prefix + "manifest.json")
