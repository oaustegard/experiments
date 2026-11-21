"""Split a pack's weights.bin into gzip shards for hosting, and optionally upload the pack to R2.

usage: shard.py PACK_DIR [--mb 64] [--upload KEY_PREFIX]
Writes PACK_DIR/host/{manifest.json, tokenizer.json, weights-NNN.bin.gz}. The hosted manifest lists
`shards: [{file, bytes}]` (bytes = uncompressed length); web/whisper.js fetches them in order. With
--upload the files go to the public R2 bucket under KEY_PREFIX, each shard stored with
Content-Encoding: gzip so the browser inflates it (token handling as kokoro-web/upload_r2.py).
"""
import argparse, gzip, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

BUCKET = "austegard-media"
PUBLIC = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev"


def put(path, key, ctype, encoding=None):
    url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}/r2/buckets/{BUCKET}/objects/{key}"
    with tempfile.NamedTemporaryFile("w", delete=False) as h:
        os.chmod(h.name, 0o600)
        h.write(f"Authorization: Bearer {os.environ['CF_API_TOKEN']}\n")
    try:
        cmd = ["curl", "-sS", "-X", "PUT", "-H", f"@{h.name}", "-H", f"Content-Type: {ctype}"]
        if encoding:
            cmd += ["-H", f"Content-Encoding: {encoding}"]
        r = subprocess.run(cmd + ["--data-binary", f"@{path}", url], capture_output=True, text=True)
    finally:
        os.unlink(h.name)
    if r.returncode != 0 or '"success":true' not in r.stdout.replace(" ", ""):
        sys.exit(f"PUT {key} failed: {r.stdout[:300]} {r.stderr[:300]}")


ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("--mb", type=int, default=64)
ap.add_argument("--upload")
a = ap.parse_args()
pack = Path(a.pack)
host = pack / "host"
if host.exists():
    shutil.rmtree(host)
host.mkdir()
m = json.loads((pack / "manifest.json").read_text())
size = a.mb * 1024 * 1024
shards, total = [], 0
with open(pack / "weights.bin", "rb") as f:
    while chunk := f.read(size):
        name = f"weights-{len(shards):03d}.bin.gz"
        with gzip.open(host / name, "wb", compresslevel=6) as g:
            g.write(chunk)
        shards.append({"file": name, "bytes": len(chunk), "gzip": (host / name).stat().st_size})
        total += shards[-1]["gzip"]
m["shards"] = shards
(host / "manifest.json").write_text(json.dumps(m))
shutil.copy(pack / "tokenizer.json", host / "tokenizer.json")
print(f"{pack.name}: {len(shards)} shards, {sum(s['bytes'] for s in shards)} bytes raw, {total} gzip", flush=True)
if a.upload:
    if not os.environ.get("CF_ACCOUNT_ID") or not os.environ.get("CF_API_TOKEN"):
        sys.exit("CF_ACCOUNT_ID / CF_API_TOKEN not set")
    for s in shards:
        put(host / s["file"], f"{a.upload}/{s['file']}", "application/octet-stream", "gzip")
        print("  up", s["file"], flush=True)
    put(host / "tokenizer.json", f"{a.upload}/tokenizer.json", "application/json")
    put(host / "manifest.json", f"{a.upload}/manifest.json", "application/json")
    print(f"{PUBLIC}/{a.upload}/manifest.json")
