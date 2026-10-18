"""Import the statically linked kernel module, time run(n), print the kernel's JSON.

usage (inside the guest): driver.py KERNEL N
The module must come from the interpreter's builtin table, not from a .py file:
the assertion below checks that.
"""
import importlib
import json
import sys
import time

kernel = sys.argv[1]
n = int(sys.argv[2])
mod = importlib.import_module(kernel)
origin = getattr(mod.__spec__, "origin", None)
print(f"driver: {kernel} origin={origin!r} run={type(mod.run).__name__}", file=sys.stderr)
assert origin == "built-in", f"expected the compiled builtin module, got origin={origin!r}"
t0 = time.perf_counter()
r = mod.run(n)
t1 = time.perf_counter()
print(json.dumps({"kernel": kernel, "n": n, "checksum": r, "seconds": t1 - t0}))
