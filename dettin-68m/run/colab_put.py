"""Upload files to /content/data on a Colab session in 18 MB parts (larger uploads return HTTP 500).

    python3 run/colab_put.py SESSION FILE [FILE ...]
A file is gzipped first unless it already ends in .gz; the VM side joins the parts, gunzips what this
script compressed, and prints each file's size and sha256 prefix, which is compared with the local one.
"""
import gzip
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PART = 18 * 1024 * 1024


def colab(*args, timeout=900):
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout, check=False)


def main():
    session, files = sys.argv[1], [Path(f) for f in sys.argv[2:]]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mk = td / "mk.py"
        mk.write_text("import os; os.makedirs('/content/data', exist_ok=True); print('ok')\n")
        print(colab("exec", "-s", session, "-f", str(mk), "--timeout", "120").stdout.strip()[-80:])
        for f in files:
            zipped = f.suffix != ".gz"
            src = f
            if zipped:
                src = td / (f.name + ".gz")
                with open(f, "rb") as i, gzip.open(src, "wb", compresslevel=6) as o:
                    shutil.copyfileobj(i, o)
            data = src.read_bytes()
            n = 0
            for off in range(0, len(data), PART):
                part = td / f"{src.name}.{n:03d}"
                part.write_bytes(data[off: off + PART])
                r = colab("upload", "-s", session, str(part), f"/content/data/{part.name}")
                if r.returncode != 0:
                    r = colab("upload", "-s", session, str(part), f"/content/data/{part.name}")
                assert r.returncode == 0, (part.name, r.stderr[-300:])
                part.unlink()
                n += 1
            join = td / "join.py"
            join.write_text(f'''
import glob, gzip, hashlib, os, shutil
parts = sorted(glob.glob("/content/data/{src.name}.[0-9][0-9][0-9]"))
with open("/content/data/{src.name}", "wb") as o:
    for p in parts:
        with open(p, "rb") as i: shutil.copyfileobj(i, o)
        os.remove(p)
final = "/content/data/{f.name}"
if {zipped!r}:
    with gzip.open("/content/data/{src.name}", "rb") as i, open(final, "wb") as o: shutil.copyfileobj(i, o)
    os.remove("/content/data/{src.name}")
h = hashlib.sha256()
with open(final, "rb") as i:
    for b in iter(lambda: i.read(1 << 20), b""): h.update(b)
print("REMOTE", os.path.getsize(final), h.hexdigest()[:16])
''')
            r = colab("exec", "-s", session, "-f", str(join), "--timeout", "600")
            local = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            line = [x for x in r.stdout.splitlines() if x.startswith("REMOTE")]
            ok = bool(line) and line[0].split()[-1] == local
            print(f"{f.name}: {n} parts, local {f.stat().st_size} {local}, {line[0] if line else r.stdout[-200:] + r.stderr[-200:]}"
                  f" -> {'match' if ok else 'MISMATCH'}", flush=True)
            assert ok, f.name


if __name__ == "__main__":
    main()
