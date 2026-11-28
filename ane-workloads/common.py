"""Shared by the three ane-workloads scripts: environment record, the mains-power guard,
a private venv, and the GPU load that the "GPU busy" condition runs under.

The GPU load is bonsai-metal-bench's own: its PrismML llama.cpp build and its PQ2_0 GGUF
(both under bonsai-metal-bench/work, built or downloaded here if that benchmark has not run
yet), driven as back-to-back `llama-bench` decode runs. Each run's tokens/s is kept, so a
result carries both sides: how much the GPU slows the Neural Engine workload and how much
the workload slows the GPU.
"""
from __future__ import annotations

import json
import os
import platform
import atexit
import contextlib
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

WORK = HERE / "work"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd, timeout=None, env=None, cwd=None):
    """Run a command; return (returncode, stdout, stderr). rc is None if it could not run."""
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", repr(e)


def sh(cmd):
    rc, out, _ = run(cmd, timeout=60)
    return out.strip() if rc == 0 else None


def is_mac():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def capture_env():
    """Chip, memory, macOS, power source and energy mode. Every field is None off macOS."""
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()}",
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": sh(["sysctl", "-n", "hw.model"]),
        "memsize_bytes": sh(["sysctl", "-n", "hw.memsize"]),
        "perf_cores": sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
        "macos": sh(["sw_vers", "-productVersion"]),
        "macos_build": sh(["sw_vers", "-buildVersion"]),
        "pmset_batt": sh(["pmset", "-g", "batt"]),
        "pmset_therm": sh(["pmset", "-g", "therm"]),
        "python": platform.python_version(),
        "low_power_mode": None,
        "on_ac_power": None,
    }
    for line in (sh(["pmset", "-g"]) or "").splitlines():
        parts = line.split()
        # lowpowermode 0/1 on most Macs; powermode 0 auto, 1 low power, 2 high power on Pro/Max
        if parts and parts[0] in ("lowpowermode", "powermode") and len(parts) > 1:
            env[parts[0]] = parts[1]
            env["low_power_mode"] = parts[1] == "1"
    if env["pmset_batt"]:
        env["on_ac_power"] = "AC Power" in env["pmset_batt"]
    return env


def require_mains(env, force):
    """Exit unless plugged in and out of Low Power Mode (or --force)."""
    if (env["on_ac_power"] is False or env["low_power_mode"]) and not force:
        sys.exit("On battery or in Low Power Mode: timings would not be comparable. Plug in and "
                 "set Energy Mode to Automatic or High Power, or pass --force.")


_LEASE = contextlib.ExitStack()
atexit.register(_LEASE.close)


def _take_gpu_lease(name):
    """Exclusive GPU lease (_lib/gpulease.py) held until the script exits: the GPU-busy condition
    is this script's own llama-bench load, and nothing else leased may share the GPU."""
    try:
        from _lib.gpulease import lease
    except ImportError:
        log("warning: _lib/gpulease.py not found; running without a GPU lease")
        return None
    held = _LEASE.enter_context(lease(f"ane-workloads/{name}", exclusive=True, on_wait=lambda st: log(
        f"waiting for the GPU: holders {[(h['name'], h['mode']) for h in st['holders']]}")))
    if held.get("unleased_gpu_processes"):
        log(f"warning: GPU processes without a lease are running: {held['unleased_gpu_processes']}")
    return held


def start(name, args, mac_only=True):
    """Common preamble: environment, power guard, results/<timestamp>-<chip>/<name>/."""
    env = capture_env()
    if mac_only and not is_mac():
        sys.exit("This script needs Apple Silicon macOS.")
    log(f"{env['chip'] or env['platform']}, macOS {env['macos']}, AC={env['on_ac_power']}, "
        f"low power={env['low_power_mode']}")
    require_mains(env, args.force)
    env["gpu_lease"] = _take_gpu_lease(name)
    tag = f"{datetime.now():%Y%m%d-%H%M}-{(env['chip'] or platform.machine()).replace(' ', '-')}"
    out = HERE / "results" / tag / name
    (out / "raw").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    (out / "env.json").write_text(json.dumps(env, indent=1))
    return env, out


def finish(out, summary, md):
    summary["env_after"] = {"pmset_therm": sh(["pmset", "-g", "therm"]),
                            "pmset_batt": sh(["pmset", "-g", "batt"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md)
    log(f"wrote {out}")


def ensure_venv(name, packages, python=None):
    """A venv under work/ with these pip requirements; returns its python."""
    vdir = WORK / f"venv-{name}"
    py = vdir / "bin" / "python"
    if not py.exists():
        log(f"creating work/venv-{name}")
        rc, _, err = run([python or sys.executable, "-m", "venv", vdir], timeout=600)
        if rc != 0:
            raise RuntimeError("venv failed: " + err[-2000:])
        run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], timeout=600)
    stamp = vdir / "installed.txt"
    want = "\n".join(packages)
    if not stamp.exists() or stamp.read_text() != want:
        log(f"pip install into work/venv-{name}: {' '.join(packages)}")
        rc, _, err = run([py, "-m", "pip", "install", "-q", *packages], timeout=3600)
        if rc != 0:
            raise RuntimeError("pip install failed: " + err[-2000:])
        stamp.write_text(want)
    return py


def reexec_in_venv(name, packages):
    """Run this same script inside its venv (once); returns when already inside it."""
    if os.environ.get("ANE_WORKLOADS_VENV") == name:
        return
    py = ensure_venv(name, packages)
    env = dict(os.environ, ANE_WORKLOADS_VENV=name)
    os.execve(str(py), [str(py), *sys.argv], env)


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ---------------------------------------------------------------- GPU load

def add_gpu_args(ap):
    ap.add_argument("--force", action="store_true",
                    help="run on battery or in Low Power Mode (results are not comparable)")
    ap.add_argument("--no-gpu-busy", action="store_true", help="skip the GPU-busy condition")
    ap.add_argument("--llama-bench", type=Path, help="llama-bench binary (default: build "
                    "bonsai-metal-bench's PrismML fork)")
    ap.add_argument("--gpu-model", type=Path, help="GGUF for the GPU load (default: "
                    "bonsai-metal-bench's Bonsai 2.5 27B PQ2_0, about 7 GB)")
    ap.add_argument("--gpu-n", type=int, default=512, help="tokens per llama-bench decode rep")
    ap.add_argument("--gpu-reps", type=int, default=6, help="reps per llama-bench run")
    ap.add_argument("--gpu-settle", type=int, default=20,
                    help="seconds after a llama-bench run starts before it counts as loading "
                    "the GPU (model load and warm-up)")


def resolve_gpu_load(args):
    """(llama-bench path, gguf path), reusing bonsai-metal-bench's build and download."""
    bench_bin, model = args.llama_bench, args.gpu_model
    if bench_bin is None or model is None:
        sys.path.insert(0, str(experiment("bonsai-metal-bench")))
        import bench  # bonsai-metal-bench/bench.py: build_fork, hf_download, llama_bench

        if bench_bin is None:
            bindir, _ = bench.build_fork()
            bench_bin = bindir / "llama-bench"
        if model is None:
            model = bench.hf_download(bench.GGUF_REPO, bench.GGUFS["PQ2_0"], bench.WORK / "gguf")
    return Path(bench_bin).expanduser().resolve(), Path(model).expanduser().resolve()


class GpuLoad:
    """Back-to-back `llama-bench -p 0 -n N -r R` decode runs on the GPU until stopped.

    llama-bench reloads the model for each run, so the GPU idles for a few seconds between
    runs; `busy_fraction(t0, t1)` reports how much of a window fell inside a run and past
    its settle time, and `rate(t0, t1)` the decode tokens/s of the runs that overlap it.
    """

    def __init__(self, bench_bin, model, n=512, reps=6, settle=20, raw=None):
        self.cmd = [str(bench_bin), "-m", str(model), "-ngl", "99", "-fa", "1", "-o", "json",
                    "-p", "0", "-n", str(n), "-r", str(reps)]
        self.settle, self.raw = settle, raw
        self.runs, self.errors = [], []
        self._stop = threading.Event()
        self._proc = None
        self._cur_start = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            self._cur_start = t0
            try:
                self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, text=True)
                out, err = self._proc.communicate()
            except OSError as e:
                self.errors.append(repr(e))
                return
            t1 = time.time()
            rec = {"t0": t0, "t1": t1, "rc": self._proc.returncode, "tg_ts": None}
            try:
                rows = json.loads(out[out.index("["):])
                rec["tg_ts"] = next((r.get("avg_ts") for r in rows if r.get("n_gen")), None)
            except ValueError:
                pass
            if self._stop.is_set():
                rec["killed"] = True
            elif rec["tg_ts"] is None:
                self.errors.append(f"rc={rec['rc']}: {err[-400:]}")
                if len(self.errors) >= 3:
                    return
            self.runs.append(rec)
            self._cur_start = None

    def start(self):
        self._thread.start()
        return self

    def baseline(self, timeout=1800):
        """Block until the first run has finished with the GPU otherwise idle; its tokens/s."""
        t = time.time()
        while not self.runs and self._thread.is_alive() and time.time() - t < timeout:
            time.sleep(1)
        if not self.runs or self.runs[0]["tg_ts"] is None:
            raise RuntimeError("GPU load did not produce a decode rate: " + "; ".join(self.errors))
        return self.runs[0]["tg_ts"]

    def wait_busy(self, timeout=300):
        """Block until a run is under way and past its settle time."""
        t = time.time()
        while time.time() - t < timeout:
            if not self._thread.is_alive():
                raise RuntimeError("GPU load stopped: " + "; ".join(self.errors))
            s = self._cur_start
            if s is not None and time.time() - s >= self.settle:
                return
            time.sleep(0.5)
        raise RuntimeError("GPU load not busy within timeout")

    def _spans(self):
        spans = [(r["t0"] + self.settle, r["t1"]) for r in self.runs]
        if self._cur_start is not None:
            spans.append((self._cur_start + self.settle, time.time()))
        return spans

    def busy_fraction(self, t0, t1):
        if t1 <= t0:
            return None
        got = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in self._spans())
        return round(got / (t1 - t0), 3)

    def rate(self, t0, t1):
        """Median decode tokens/s over completed runs that overlap [t0, t1]."""
        return median([r["tg_ts"] for r in self.runs
                       if r["tg_ts"] and not r.get("killed") and r["t1"] > t0 and r["t0"] < t1])

    def stop(self):
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._thread.join(timeout=30)
        if self.raw is not None:
            Path(self.raw).write_text(json.dumps({"cmd": self.cmd, "runs": self.runs,
                                                  "errors": self.errors}, indent=1))


def gpu_conditions(args, raw):
    """Yield ("idle", None) then ("busy", GpuLoad) unless --no-gpu-busy; a load that cannot
    start yields ("busy", exception) so the caller records why and carries on."""
    yield "idle", None
    if args.no_gpu_busy:
        return
    try:
        bench_bin, model = resolve_gpu_load(args)
        load = GpuLoad(bench_bin, model, args.gpu_n, args.gpu_reps, args.gpu_settle,
                       raw / "gpu-load.json").start()
        log("GPU load: first llama-bench run alone (the GPU's own baseline)")
        load.alone_ts = load.baseline()
        log(f"GPU alone: {load.alone_ts:.1f} tok/s decode")
    except Exception as e:  # noqa: BLE001 - no GPU load must not lose the idle result
        yield "busy", e
        return
    try:
        load.wait_busy()
        yield "busy", load
    finally:
        load.stop()


def gpu_record(load, t0, t1):
    """What the GPU did during [t0, t1] of a busy-condition measurement."""
    if load is None:
        return {}
    during = load.rate(t0, t1)
    return {"gpu_busy_fraction": load.busy_fraction(t0, t1),
            "gpu_alone_tok_s": round(load.alone_ts, 2),
            "gpu_during_tok_s": round(during, 2) if during else None}
