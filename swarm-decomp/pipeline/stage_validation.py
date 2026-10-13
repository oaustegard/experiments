#!/usr/bin/env python3
"""Stage a validation tree in the jail: the runner plus the ORIGINAL sources as the
'reconstruction'. The runner then runs there under jail.sh; every function must PASS,
which checks the comparator and the TU generator end to end. Copies files only.

    stage_validation.py O2      # then: jail.sh -C <jail>/validate -e DECOMP_EXTRA_I=<jail>/validate/inc -- python3 tests/run.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JAIL = Path("/tmp/jail")
opt = sys.argv[1] if len(sys.argv) > 1 else "O2"
V = JAIL / "validate"
if V.exists():
    shutil.rmtree(V)
shutil.copytree(ROOT / "tests" / "lib", V / "tests" / "lib", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(ROOT / "tests" / "run.py", V / "tests" / "run.py")
fl = json.loads((ROOT / "tests" / "flags.json").read_text())
fl["opt"] = "-" + (sys.argv[2] if len(sys.argv) > 2 else opt)   # a different level = a negative control
(V / "tests" / "flags.json").write_text(json.dumps(fl))
shutil.copytree(ROOT / "build" / f"suites-{opt}", V / "tests" / "suites")
shutil.copytree(ROOT / "tests" / "include", V / "tests" / "include")
T = json.loads((ROOT / "pipeline" / "targets.json").read_text())
for t in T.values():
    for pname, p in t["projects"].items():
        sd = JAIL / "src" / (p.get("subdir") or pname)
        (V / "inc" / pname).mkdir(parents=True, exist_ok=True)
        for h in sd.glob("*.h"):
            shutil.copy(h, V / "inc" / pname / h.name)
        for f in p["files"]:
            d = V / "src" / pname / f
            d.mkdir(parents=True, exist_ok=True)
            # the project's own -D flags belong to the original build: replay them as #defines
            defs = ""
            if pname != "lua":
                for x in p.get("defines", []):
                    k, _, v = x[2:].partition("=")
                    defs += f"#define {k} {v or '1'}\n"
            src = (sd / f"{f}.c").read_bytes()
            (d / f"{f}.c").write_bytes(defs.encode() + src)
print("staged", V)
