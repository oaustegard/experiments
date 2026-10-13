#!/usr/bin/env python3
"""Re-download the targets' original source for analysis, AFTER every worker has stopped.

    fetch_upstream.py [--out build/upstream]     -> build/upstream/<proj>/*.c *.h

Neat projects: `git clone` at the recorded commit, checked against the recorded sha256 of
`git archive --format=tar <commit>`. Lua: the lua.org tarball, checked against its sha256.
Only the files are copied out; nothing is built or run. build/ is gitignored, so the source
never lands in the experiment repo."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = json.loads((ROOT / "pipeline" / "targets.json").read_text())
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "build" / "upstream"))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for t in T.values():
        for pname, p in t["projects"].items():
            dest = out / pname
            dest.mkdir(exist_ok=True)
            if "commit" in p:
                d = td / pname
                subprocess.run(["git", "clone", "-q", p["url"], str(d)], check=True)
                subprocess.run(["git", "-C", str(d), "checkout", "-q", p["commit"]], check=True)
                tar = subprocess.run(["git", "-C", str(d), "archive", "--format=tar", p["commit"]],
                                     capture_output=True, check=True).stdout
                got = hashlib.sha256(tar).hexdigest()
                src = d
            else:
                tgz = td / Path(p["url"]).name
                urllib.request.urlretrieve(p["url"], tgz)
                got = hashlib.sha256(tgz.read_bytes()).hexdigest()
                with tarfile.open(tgz) as tf:
                    tf.extractall(td, filter="data")
                src = td / p["subdir"]
            ok = got == p["sha256"]
            print(f"{pname}: sha256 {'ok' if ok else 'MISMATCH ' + got}")
            for f in list(src.glob("*.c")) + list(src.glob("*.h")):
                shutil.copy(f, dest / f.name)
