"""Create an agent working directory: runs/<run>/<target>/ with the task,
the Mojo notes, a building starter package, build.sh and check.sh.

    python3 harness/new_run.py r1 snowball difflib yake
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = {"snowball": "msnowball", "difflib": "mdifflib", "yake": "myake"}
ORACLE = {"snowball": "oracle/snowball/check.py", "difflib": "oracle/difflib/fuzz.py", "yake": "oracle/yake/check.py"}

KERNEL = '''from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer


def sum_u32(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Example: sum a numpy uint32 buffer given its address and length."""
    var p = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(py=addr))
    var total = 0
    for i in range(Int(py=n)):
        total += Int(p[unsafe_offset=i])
    return PythonObject(total)


@export
def PyInit__kernel() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_kernel")
        m.def_function[sum_u32]("sum_u32", docstring="example")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
'''

INIT = '''"""Python shim over the Mojo kernel in _kernel.so (built by build.sh)."""
from . import _kernel  # noqa: F401
'''


def main() -> int:
    run, targets = sys.argv[1], sys.argv[2:]
    for target in targets:
        pkg = PKG[target]
        work = ROOT / "runs" / run / target
        (work / pkg).mkdir(parents=True, exist_ok=False)
        task = (ROOT / "kit" / f"TASK_{target}.md").read_text()
        (work / "TASK.md").write_text(task.replace("BENCH", str(ROOT / "bench/bench.py")))
        (work / "MOJO_NOTES.md").write_text((ROOT / "kit" / "MOJO_NOTES.md").read_text())
        (work / pkg / "kernel.mojo").write_text(KERNEL)
        (work / pkg / "__init__.py").write_text(INIT)
        (work / "build.sh").write_text(
            "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\n"
            f"mojo build {pkg}/kernel.mojo --emit shared-lib -o {pkg}/_kernel.so\n")
        (work / "check.sh").write_text(
            "#!/bin/sh\ncd \"$(dirname \"$0\")\"\nsh build.sh || exit 1\n"
            f"python3 {ROOT / ORACLE[target]} --pkg . --split dev \"$@\"\n")
        print(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
