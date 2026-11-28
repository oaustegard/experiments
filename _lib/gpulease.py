"""One GPU, many jobs: a lease every GPU job on the Mac takes, so measurements are not contended.

Round 1 of bonsai-metal-bench and the inkling-fact-forwarding lane tests ran on the same M5 GPU at
the same time without knowing it: a `llama-bench` cut Inkling's single-stream decode from ~100 to
~11 tok/s. This module makes GPU use explicit.

* An **exclusive** lease (benchmarks, timing runs) waits until no one else holds the GPU, and while
  it waits it posts a drain request so shared holders stop taking new work.
* A **shared** lease (a serving process, an MCP batch call) coexists with other shared holders and
  is refused or delayed while an exclusive holder runs or waits.
* Every holder writes a record (pid, name, mode, since), so `status` says who is on the GPU, and it
  also lists GPU-heavy processes that hold no lease (llama-bench, inkling, mlx_lm, llama-server).

Mechanism: fcntl.flock on one lock file (LOCK_SH / LOCK_EX), released by the kernel when the
process dies, plus small JSON files for holders and drain requests. Stdlib only; macOS and Linux.

    from _lib.gpulease import lease
    with lease("bonsai-bench", exclusive=True):
        ...                                  # nothing else leased is on the GPU here

    python3 -m _lib.gpulease status
    python3 -m _lib.gpulease run --exclusive --name prefill-ab -- ./llama-bench ...
    python3 -m _lib.gpulease run --shared --name inkling-batch -- ./inkling batch-json ...

A shared holder that runs long (a resident server) should call `drain_requested()` between units
of work and release when it returns a request. Set GPULEASE=0 to bypass the lease entirely.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(os.environ.get("GPULEASE_DIR") or Path.home() / ".gpulease")
LOCK = DIR / "gpu.lock"
HOLDERS = DIR / "holders"
DRAIN = DIR / "drain"
GPU_PROCS = ("llama-bench", "llama-server", "llama-cli", "inkling", "mlx_lm", "mlx-lm",
             "whisper-cli", "whisperkit-cli", "ollama")


class LeaseTimeout(TimeoutError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _records(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if _alive(int(r.get("pid", -1))):
            out.append(r)
        else:
            p.unlink(missing_ok=True)   # stale: the process died without cleaning up
    return out


def drain_requested() -> dict | None:
    """The oldest live drain request, or None. Shared holders check this between work units."""
    reqs = _records(DRAIN)
    return min(reqs, key=lambda r: r["since"]) if reqs else None


def unleased_gpu_processes() -> list[str] | None:
    """GPU-heavy processes with no lease record. None if ps cannot run (unknown, not 'none')."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,pcpu=,command="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    leased = {int(r["pid"]) for r in _records(HOLDERS)}
    mine = {os.getpid(), os.getppid()}
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[2]
        exe = os.path.basename(cmd.split()[0]) if cmd.split() else ""
        if pid in mine or pid in leased or "gpulease" in cmd:
            continue
        if any(exe.startswith(g) for g in GPU_PROCS) or any(f"-m {g}" in cmd for g in ("mlx_lm",)):
            found.append(f"{pid} {parts[1]}% {cmd[:120]}")
    return found


def status() -> dict:
    return {"dir": str(DIR), "holders": _records(HOLDERS), "drain_requests": _records(DRAIN),
            "unleased_gpu_processes": unleased_gpu_processes()}


@contextlib.contextmanager
def lease(name: str, exclusive: bool = False, timeout: float | None = None, poll: float = 0.5,
          on_wait=None):
    """Hold the GPU lease for the duration of the block.

    exclusive: wait for every other holder to leave, posting a drain request meanwhile.
    shared: wait while an exclusive holder runs or a drain request is pending.
    timeout: seconds to wait before LeaseTimeout (None waits forever).
    on_wait(status_dict): called about every 10 s while waiting, for progress messages.
    """
    if os.environ.get("GPULEASE") == "0":
        yield {"bypassed": True}
        return
    for d in (HOLDERS, DRAIN):
        d.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    me = {"pid": os.getpid(), "name": name, "mode": "exclusive" if exclusive else "shared",
          "since": time.time(), "cmd": " ".join(sys.argv)[:200]}
    drain_file = DRAIN / f"{os.getpid()}.json"
    holder_file = HOLDERS / f"{os.getpid()}.json"
    t0, last_note = time.time(), 0.0
    try:
        while True:
            if exclusive:
                drain_file.write_text(json.dumps({**me, "reason": f"exclusive lease for {name}"}))
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            else:
                pending = [r for r in _records(DRAIN) if r["pid"] != os.getpid()]
                if not pending:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        # an exclusive request may have landed between the check and the lock
                        if not [r for r in _records(DRAIN) if r["pid"] != os.getpid()]:
                            break
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except BlockingIOError:
                        pass
            if timeout is not None and time.time() - t0 > timeout:
                raise LeaseTimeout(f"no {me['mode']} GPU lease for {name} after {timeout:.0f} s: {status()}")
            if on_wait and time.time() - last_note > 10:
                on_wait(status())
                last_note = time.time()
            time.sleep(poll)
        drain_file.unlink(missing_ok=True)
        me["acquired"] = time.time()
        me["waited_s"] = round(me["acquired"] - t0, 2)
        holder_file.write_text(json.dumps(me))
        yield {**me, "unleased_gpu_processes": unleased_gpu_processes()}
    finally:
        holder_file.unlink(missing_ok=True)
        drain_file.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _main() -> int:
    ap = argparse.ArgumentParser(prog="python3 -m _lib.gpulease", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="who holds the GPU, pending drains, unleased GPU processes")
    r = sub.add_parser("run", help="run a command under a lease")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--exclusive", action="store_true")
    g.add_argument("--shared", action="store_true")
    r.add_argument("--name", required=True)
    r.add_argument("--timeout", type=float, default=None)
    r.add_argument("argv", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(status(), indent=1))
        return 0
    argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
    if not argv:
        ap.error("run needs a command after --")
    note = lambda s: print(f"gpulease: waiting for {'exclusive' if a.exclusive else 'shared'} lease; "
                           f"holders={[(h['name'], h['mode']) for h in s['holders']]} "
                           f"drains={[d['name'] for d in s['drain_requests']]}", file=sys.stderr)
    try:
        with lease(a.name, exclusive=a.exclusive, timeout=a.timeout, on_wait=note) as info:
            if info.get("unleased_gpu_processes"):
                print(f"gpulease: warning, GPU processes without a lease: {info['unleased_gpu_processes']}",
                      file=sys.stderr)
            return subprocess.call(argv)
    except LeaseTimeout as e:
        print(f"gpulease: {e}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    sys.exit(_main())
