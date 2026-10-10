# Mojo 1.1 notes for this port

Verified in this container on 2026-10-10 with `mojo --version` = 1.1.0.

## Your Mojo knowledge is out of date

Mojo changed a lot through 1.0 and 1.1. Code in the style you remember
(`fn`, `let`, `inout`, `UnsafePointer`, `from python import`, bare
`from math import`) fails or warns. **Before writing any Mojo, read these two
files in full** (Modular's own, current as of 2026-10-08):

- `/tmp/modskills/skills/mojo-syntax/SKILL.md` (and its `references/` when a
  topic needs depth)
- `/tmp/modskills/skills/mojo-python-interop/SKILL.md`

When the compiler and your memory disagree, the compiler is right. Read the
error, then grep those files for the construct.

## The starter files build and import

`<pkg>/kernel.mojo` builds to `<pkg>/_kernel.so`, which Python imports as
`<pkg>._kernel` (the init function must be named `PyInit__kernel`, two
underscores, because CPython looks up `PyInit_<last segment of the module
name>`). Spellings that differ from older Mojo, all checked here:

- `from std.python import PythonObject, Python`,
  `from std.python.bindings import PythonModuleBuilder`, `from std.os import abort`
- `@export def PyInit__kernel() abi("C") -> PythonObject:` (the `abi("C")` is
  required; it cannot `raises`, so wrap in `try` and `abort`)
- Registered functions take and return `PythonObject`, may `raises`, and take
  at most 6 arguments.
- Raw pointer from a numpy array:
  `Pointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(py=arr.__array_interface__["data"][0]))`,
  indexed as `p[unsafe_offset=i]` (plain `p[i]` is deprecated).
- `Int(py=obj)`, `String(py=obj)`, `Float64(py=obj)` convert from Python;
  `PythonObject(x)` converts back.

Build: `sh build.sh`. Compile errors print with file:line:col.

## Crossing the boundary is expensive, so cross it rarely

Measured here:

| operation | cost |
|---|---|
| one call into a Mojo function with two int args | ~3.3 µs |
| one str crossing in and one out (`String(py=w)` → `PythonObject(s)`) | ~1.5 µs per string, 13× slower than Python's own `str.upper` |
| `"\n".join(words).encode("utf-32-le")` plus decode and split, Python side | ~0.19 µs per word |

So do not walk Python lists or dicts element by element from Mojo in a hot
path, and do not call into Mojo once per small item if you can batch. Move
bulk data as contiguous buffers: numpy arrays (or `np.frombuffer(some_bytes,
dtype=np.uint32)`) in and out, passing only their addresses and lengths.
For text, UTF-32-LE gives one `UInt32` per Python code point, which is
exactly how Python indexes a `str`. Allocate output buffers in Python (numpy)
and have Mojo fill them.

A thin Python shim in `<pkg>/__init__.py` that reshapes inputs and outputs
is expected. The algorithm itself has to run in Mojo.

## Rules

- Work only inside your working directory. Read anything you need elsewhere.
- Your package must not import or call the library you are porting at
  runtime (no fallback to the Python implementation). Exceptions are named
  in TASK.md.
- `sh check.sh` runs the correctness check on the dev split. The final grade
  uses a held-out split you cannot see, rebuilt from your sources with your
  `build.sh`, so make the code general rather than fitting the dev cases.
- Do not edit anything outside your working directory, including the oracle.
