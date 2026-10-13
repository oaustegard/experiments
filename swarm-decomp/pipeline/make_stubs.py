#!/usr/bin/env python3
"""Initial state: one stub file per target function, so every function starts failing.

    make_stubs.py [--only neat|lua]

src/<proj>/<tu>/<function>.c holds only a TODO(decomp) comment, which the runner treats as
"not yet written" (stage missing, reason "stub: ..."). gcc clones (foo.part.0, foo.isra.0,
foo.constprop.0) map to their base function's file. Existing non-stub files are never touched.
No decompiler draft is seeded: workers start from the disassembly alone."""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "lib"))
import objcmp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
a = ap.parse_args()
n = 0
for sd in sorted((ROOT / "tests" / "suites").iterdir()):
    proj, _, tu = sd.name.partition("-")
    if a.only and not (proj == a.only or proj.startswith(a.only)):
        continue
    obj = objcmp.Obj(str(sd / "target.o"))
    d = ROOT / "src" / proj / tu
    d.mkdir(parents=True, exist_ok=True)
    (ROOT / "src" / proj / "types").mkdir(parents=True, exist_ok=True)
    keep = ROOT / "src" / proj / "types" / ".keep"
    keep.touch()
    for name in obj.order:
        base = re.sub(r"\.(part|isra|constprop|cold)\.\d+.*$", "", name)
        f = d / f"{base}.c"
        if f.exists():
            continue
        f.write_text(f"/* TODO(decomp): {base} -- see tests/suites/{sd.name}/{name} */\n")
        n += 1
print(f"{n} stubs written")
