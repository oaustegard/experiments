# SPy arm: status notes

**Status: blocked before any kernel was ported or built. Nothing here is verified.**

## SPy source

- Repo: https://github.com/spylang/spy
- Commit: `a39950cab36e7adeca4808fb0e75081526ee44c8` (2026-10-12, "Introduce the SIMD type (#734)")
- Cloned shallow with `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1` into `/tmp/jail/spy`, then `chown -R nobody:nogroup`.
- Probe through the jail (`sh scripts/jail.sh -C /tmp/jail/spy -- ...`) works: uid 65534, repo readable and writable, `/opt/pwb/wasi-sdk-21/bin` readable, `/usr/bin/python3.12` present (3.12.3).

## Blocker

Installing SPy's runtime dependencies into `/tmp/jail/spy/.venv` was refused by the `jail_route` PreToolUse hook. The hook blocks any command that names `/tmp/jail` and runs a non-jail interpreter (`python3.12 -m venv`, and the same for `timeout`). The install was the only step that needs network outside the jail. I did not work around the hook.

No system interpreter has the dependencies. Checked with `/usr/bin/python3.12`, `python3.13` and `python3.11`: `click` is present only in 3.13, and `py` only in 3.13. The rest are missing (`typer`, `fixedint`, `pexpect`, `wasmtime`, `ziglang`, `ninja`, `build`).

Pinned runtime dependencies from `pyproject.toml` (requires-python `~=3.12.0`):
`build~=1.3`, `click==8.1.8`, `fixedint==0.2.0`, `ninja==1.13.2`, `py==1.11.0`, `typer==0.15.1`, `pexpect==4.9.0`, `wasmtime==48.0.0`, `ziglang==0.13.0`.

Install command to run once the hook allows it (wheels only, so no sdist build backend runs):
`python3.12 -m venv /tmp/jail/spy/.venv && /tmp/jail/spy/.venv/bin/python -m pip install --only-binary=:all: <the pins above>`, then `chown -R nobody:nogroup /tmp/jail/spy`.
Run the compiler with `sh /home/user/claude-workspace/scripts/jail.sh -C /tmp/jail/spy -e PYTHONPATH=/tmp/jail/spy -- python -m spy.cli ...`.

## Build route (from reading the source)

- The wasi target does not use wasi-sdk. `spy/build/flags.py` sets `CC = python -m ziglang cc` and `--target=wasm32-wasi-musl`. The C backend is driven from `spy/build/config.py`. So `ziglang==0.13.0` is the toolchain, and wasi-sdk 21 is not needed for SPy's own build.
- `spy build` with `--target=wasi` and `--output-kind=exe` produces a `.wasm`. `spy/libspy/Makefile` builds libspy for wasi first (`make TARGET=wasi`), and the tests use `spy/libspy/build/...`.
- Execution: `spy/llwasm/wasmtime.py` runs wasm in-process via the `wasmtime` Python package, and `spy/tool/wasmtime.py` runs WASI modules. For `run.sh` the plan is `tools/run_wasi.sh` with `RUNTIME=wasmtime` (the wasmtime CLI 39.0.1 is under `/opt/pwb`).

## Dialect facts checked in the source (not yet compiled)

- `from math import sqrt`, `pi`, `tau`, `e`, plus `log`, `exp`, `sin`, `cos`, and so on (`spy/vm/modules/math.py`). The `** 0.5` in the Python kernels maps to `x ** 0.5` on f64, which is supported (`spy/tests/compiler/test_float.py::test_pow`, backed by C `pow`). A `sqrt` port is the alternative, but it changes the operation, so `**` is the closer match.
- `int` is an alias for `i32` (`examples/1_high_level/fibo.spy`). The LCG in `dictwords.py` and the `mandelbrot` checksum need 64-bit arithmetic, so those must use `i64` explicitly. `i64` exists as a builtin type (`spy/vm/primitive.py`).
- `time`: `from time import time` gives `time() -> f64` (`spy/vm/modules/time.py`). Timing around the kernel only is therefore possible.
- Argv: `def main(argv: list[str]) -> i32` is supported (`spy/tests/compiler/test_main_func.py`). N can be read from `argv[1]` and parsed, but the int parse path is unverified.
- `str(f64)` exists. `f64.__format__` only supports the empty spec (`stdlib/_format.spy`). The JSON line needs a float printed with enough digits for a 1e-9 relative check, which is unverified.
- Lists (`stdlib/_list.spy`: `append`, `pop`), dicts (`stdlib/_dict.spy`: `get(key, default)`), and `str.join` are in the stdlib.
- `sorted` is not in the stdlib, and no `key=` lambda sort was found. `dictwords` depends on `sorted(counts.items(), key=...)`, so it is likely inexpressible as written. A hand-written sort with the same ordering `(-count, word)` would be a port, not a fake, but it has to be checked against the Python checksum.

## Kernel status

| kernel | status |
|---|---|
| nbody | not ported (blocked) |
| mandelbrot | not ported (blocked) |
| matmul | not ported (blocked) |
| dictwords | not ported; expected to need a hand-written sort, unverified |

`run.sh` and `build.sh` were not written, because nothing could be compiled to check them.

## Next steps once the install is allowed

1. Create the venv and install the pins as above, then `chown -R nobody:nogroup /tmp/jail/spy`.
2. Build libspy for wasi and compile a hello-world `.spy` through the jail. This checks the toolchain before any kernel work.
3. Port nbody, mandelbrot, matmul to `arms/spy/src/<kernel>.spy`, keeping the float operation order. Check `** 0.5` and the output precision.
4. Check dictwords against the sort and dict subset. Mark it `unsupported` (exit 3) if it cannot be expressed.
5. Verify at N = nbody 1000, mandelbrot 60, matmul 30, dictwords 5000 against `python3.11 kernels/<kernel>.py N`, with relative tolerance 1e-9.
