"""Upload a pre-compressed file to the public R2 bucket austegard-media with its Content-Encoding, so browsers
decompress it themselves (R2 does not compress .onnx on the fly). Token handling follows claude-workspace's
scripts/embstore.py: the token goes in a 0600 header file, never in argv or output.

    python upload_r2.py LOCAL_FILE KEY CONTENT_TYPE [CONTENT_ENCODING]
"""
import os
import subprocess
import sys
import tempfile

BUCKET = "austegard-media"
PUBLIC = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev"


def put(path, key, ctype, encoding=None):
    if not os.environ.get("CF_ACCOUNT_ID") or not os.environ.get("CF_API_TOKEN"):
        sys.exit("CF_ACCOUNT_ID / CF_API_TOKEN not set")
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
    print(f"{PUBLIC}/{key}")


if __name__ == "__main__":
    put(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else None)
