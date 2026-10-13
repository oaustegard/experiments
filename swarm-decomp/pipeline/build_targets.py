#!/usr/bin/env python3
"""Build the target objects and the per-TU test suites.

    build_targets.py prepare                 # sources into the jail, owned by nobody
    build_targets.py compile --opt -O2       # gcc in the jail -> build/objs/O2/<proj>/<tu>.o
    build_targets.py suites --opt -O2 [--out tests/suites] [--only neat|lua]
    build_targets.py headers                 # Lua's installed headers -> tests/include/lua

Third-party code (the targets' sources) is only ever compiled, and only inside
claude-workspace scripts/jail.sh (no network, runs as nobody, empty env). No
target binary is linked or run. Workers get the objects (symbol table kept, no
DWARF), per-function listings and public headers; never the source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "lib"))
import objcmp  # noqa: E402

JAIL = Path("/tmp/jail")
JAIL_SH = Path("/home/user/claude-workspace/scripts/jail.sh")
TARGETS = json.loads((ROOT / "pipeline" / "targets.json").read_text())
FLAGS = json.loads((ROOT / "tests" / "flags.json").read_text())
DL = Path(__file__).resolve().parent.parent / "build" / "dl"


def projects(only: str | None = None):
    for tname, t in TARGETS.items():
        if only and tname != only:
            continue
        for pname, p in t["projects"].items():
            yield tname, pname, p


def srcdir(pname: str, p: dict) -> Path:
    if "subdir" in p:
        return JAIL / "src" / p["subdir"]
    return JAIL / "src" / pname


def prepare(a) -> None:
    (JAIL / "src").mkdir(parents=True, exist_ok=True)
    for _, pname, p in projects(a.only):
        d = srcdir(pname, p)
        if "commit" in p:
            if not d.exists():
                subprocess.run(["git", "clone", "-q", p["url"], str(d)], check=True)
            subprocess.run(["git", "-C", str(d), "checkout", "-q", p["commit"]], check=True)
            head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            assert head == p["commit"], (pname, head)
        else:
            tgz = DL / Path(p["url"]).name
            if not tgz.exists():
                DL.mkdir(parents=True, exist_ok=True)
                subprocess.run(["curl", "-sSfL", "-o", str(tgz), p["url"]], check=True)
            got = subprocess.run(["sha256sum", str(tgz)], capture_output=True, text=True).stdout.split()[0]
            assert got == p["sha256"], (pname, got)
            with tarfile.open(tgz) as tf:
                tf.extractall(JAIL / "src", filter="data")
    subprocess.run(["chown", "-R", "nobody:nogroup", str(JAIL)], check=True)
    print("prepared", [pn for _, pn, _ in projects(a.only)])


def cflags(opt: str, pname: str) -> list[str]:
    return [opt, *FLAGS["common"], "-w"]


def compile_(a) -> None:
    out_root = JAIL / "out" / a.opt.lstrip("-")
    for tname, pname, p in projects(a.only):
        d = srcdir(pname, p)
        od = out_root / pname
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run(["chown", "-R", "nobody:nogroup", str(JAIL / "out")], check=True)
        dest = ROOT / "build" / "objs" / a.opt.lstrip("-") / pname
        dest.mkdir(parents=True, exist_ok=True)
        for f in p["files"]:
            cmd = ["sh", str(JAIL_SH), "-C", str(d), "--", FLAGS["cc"], "-c", *cflags(a.opt, pname),
                   *p.get("defines", []), f"{f}.c", "-o", str(od / f"{f}.o")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {pname}/{f}: {r.stderr[-400:]}")
                continue
            shutil.copy(od / f"{f}.o", dest / f"{f}.o")
            subprocess.run(["objcopy", "--strip-debug", str(dest / f"{f}.o")], check=True)
        print(f"{pname}: {len(list(dest.glob('*.o')))} objects at {a.opt}")


def headers(a) -> None:
    for _, pname, p in projects("lua"):
        dest = ROOT / "tests" / "include" / pname
        dest.mkdir(parents=True, exist_ok=True)
        for h in p.get("public_headers", []):
            shutil.copy(srcdir(pname, p) / h, dest / h)
        print("headers ->", dest)


def info_text(obj: objcmp.Obj, suite: str, pname: str, tu: str) -> str:
    lines = [f"# {suite}: translation unit {tu} of {pname}",
             f"# write each function as src/{pname}/{tu}/<function>.c (see CONTRACT.md)", "",
             "## functions, in definition order (static = file-local)"]
    for n in obj.order:
        f = obj.funcs[n]
        lines.append(f"{n}\t{'static' if f.bind == 'STB_LOCAL' else 'global'}\t{f.size} bytes")
    lines += ["", "## data objects defined in this TU (name, linkage, size, section; contents)"]
    for sym in obj.syms:
        if sym["st_info"]["type"] != "STT_OBJECT" or not isinstance(sym["st_shndx"], int):
            continue
        sec = obj.secs[sym["st_shndx"]]
        bind = "static" if sym["st_info"]["bind"] == "STB_LOCAL" else "global"
        lines.append(f"{sym.name}\t{bind}\t{sym['st_size']} bytes\t{sec.name}")
        if sec["sh_type"] == "SHT_NOBITS":
            lines.append("    (zero-initialised)")
            continue
        raw = obj.data(sym["st_shndx"])[sym["st_value"]:sym["st_value"] + sym["st_size"]]
        rels = sorted(r for r in obj.rels.get(sym["st_shndx"], [])
                      if sym["st_value"] <= r[0] < sym["st_value"] + sym["st_size"])
        if rels:
            for off, typ, si, add in rels[:200]:
                lines.append(f"    +{off - sym['st_value']:<5d} -> {obj.describe(si, add)}")
            if len(rels) > 200:
                lines.append(f"    ... {len(rels) - 200} more pointers")
        shown = raw[:512]
        for i in range(0, len(shown), 32):
            lines.append(f"    {i:5d}: {shown[i:i + 32].hex(' ')}")
        if len(raw) > 512:
            lines.append(f"    ... {len(raw) - 512} more bytes")
    undef = sorted({s.name for s in obj.syms if s["st_shndx"] == "SHN_UNDEF" and s.name})
    common = sorted({s.name for s in obj.syms if s["st_shndx"] == "SHN_COMMON"})
    lines += ["", "## symbols this TU uses but does not define (other TUs, libc)", " ".join(undef)]
    if common:
        lines += ["", "## common (tentative) definitions", " ".join(common)]
    return "\n".join(lines) + "\n"


def suites(a) -> None:
    out = ROOT / a.out
    total = 0
    for tname, pname, p in projects(a.only):
        objs = ROOT / "build" / "objs" / a.opt.lstrip("-") / pname
        for f in p["files"]:
            o = objs / f"{f}.o"
            if not o.exists():
                print("missing object", o)
                continue
            suite = f"{pname}-{f}"
            sd = out / suite
            if sd.exists():
                shutil.rmtree(sd)
            sd.mkdir(parents=True)
            shutil.copy(o, sd / "target.o")
            obj = objcmp.Obj(str(sd / "target.o"))
            for n in obj.funcs:
                (sd / n).write_text(objcmp.listing(obj, n))
            (sd / "_info.txt").write_text(info_text(obj, suite, pname, f))
            total += len(obj.funcs)
            print(f"{suite}: {len(obj.funcs)} functions")
    print("total functions:", total)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "compile", "suites", "headers"):
        s = sub.add_parser(name)
        s.add_argument("--only", default=None)
        s.add_argument("--opt", default=FLAGS["opt"])
        s.add_argument("--out", default="tests/suites")
    a = ap.parse_args()
    {"prepare": prepare, "compile": compile_, "suites": suites, "headers": headers}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
