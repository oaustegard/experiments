"""Host-side cost of server/torch_engine.py recognize(), before and after the batched window
change, on CPU.

    python3 run/bench_torch_host.py

"before" is the engine at git HEAD (server/torch_engine.py), loaded from `git show` into a
temporary module; "after" is the working-tree engine. Both run the same engine/model:
models/best/eng.traineddata on the eng lines of /tmp/benchlines/sample.jsonl, repeated to
1,024 lines, in exact and cudnn mode, torch.set_num_threads(4).

For each mode and engine it reports
  * wall lines/s (best of REPS unprofiled runs), and
  * host ms per line: a cProfile run of recognize() where every entry that is not a torch
    function or torch kernel counts (engine, tessgpu and NumPy/PIL Python and calls);
    torch.lstm and matmul/conv kernels are excluded.
Outputs of before and after are compared line by line.
"""

import cProfile
import io
import json
import pstats
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from server.torch_engine import TorchTess  # noqa: E402  (the working-tree engine)

SAMPLE = Path("/tmp/benchlines/sample.jsonl")
MODEL = str(HERE / "models/best/eng.traineddata")
N_LINES = 1024
THREADS = 4
REPS = 2
MODES = ("exact", "cudnn")


def load_head_engine():
    """server/torch_engine.py as committed at HEAD, executed as a temporary module."""
    top = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                         check=True, capture_output=True, text=True).stdout.strip()
    rel = (HERE / "server/torch_engine.py").relative_to(top).as_posix()
    src = subprocess.run(["git", "-C", top, "show", f"HEAD:{rel}"],
                         check=True, capture_output=True, text=True).stdout
    mod = types.ModuleType("torch_engine_head")
    mod.__file__ = f"HEAD:{rel}"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def load_lines():
    rows = [json.loads(l) for l in SAMPLE.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["lang"] == "eng"]
    rows = [rows[i % len(rows)] for i in range(N_LINES)]
    return [np.asarray(Image.open(r["path"]).convert("L")) for r in rows]


def timed(eng, imgs):
    best = float("inf")
    out = None
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = eng.recognize(imgs)
        best = min(best, time.perf_counter() - t0)
    return best, out


def is_torch(entry):
    """True for a profiled entry that is torch code or a torch kernel."""
    filename, _, name = entry
    return "torch" in filename or "torch" in name


def host_seconds(eng, imgs):
    """One profiled recognize() call. Returns
    host:   self time of all non-torch entries (engine, tessgpu, NumPy, PIL),
    engine: self time of the engine module's own functions alone,
    cum:    cumulative time of the engine's and tessgpu's functions, for the top rows."""
    prof = cProfile.Profile()
    prof.enable()
    eng.recognize(imgs)
    prof.disable()
    st = pstats.Stats(prof, stream=io.StringIO())
    host = engine = 0.0
    engine_cum = {}
    for entry, (cc, nc, tt, ct, callers) in st.stats.items():
        if not is_torch(entry):
            host += tt
            if entry[0].endswith(("torch_engine.py", "torch_engine_head.py")):
                engine += tt
        fn = entry[2]
        if entry[0].endswith(("torch_engine.py", "torch_engine_head.py")) or \
                entry[0].endswith(("tessgpu/ref.py", "tessgpu/decode.py")):
            engine_cum[fn] = max(engine_cum.get(fn, 0.0), ct)
    return host, engine, engine_cum


def main():
    torch.set_num_threads(THREADS)
    head = load_head_engine()
    imgs = load_lines()
    n = len(imgs)
    report = {"lines": n, "threads": THREADS, "model": "models/best/eng.traineddata",
              "modes": {}}
    print(f"{n} eng lines from {SAMPLE}, torch {torch.__version__}, {THREADS} threads")
    for mode in MODES:
        before = head.TorchTess(MODEL, device="cpu", mode=mode)
        after = TorchTess(MODEL, device="cpu", mode=mode)
        before.recognize(imgs[:16])                    # warm-up
        after.recognize(imgs[:16])
        tb, out_b = timed(before, imgs)
        ta, out_a = timed(after, imgs)
        hb, eb, cb = host_seconds(before, imgs)
        ha, ea, ca = host_seconds(after, imgs)
        diffs = [float(np.abs(p - q).max()) if p.size else 0.0 for p, q in zip(out_b, out_a)]
        same_shape = all(p.shape == q.shape for p, q in zip(out_b, out_a))
        max_diff = max(diffs)
        entry = {
            "max_abs_diff_probs": max_diff, "same_shapes": same_shape,
            "identical": bool(same_shape and max_diff == 0.0),
            "wall_lines_per_s": {"before": n / tb, "after": n / ta},
            "wall_ms_per_line": {"before": 1000 * tb / n, "after": 1000 * ta / n},
            "host_ms_per_line": {"before": 1000 * hb / n, "after": 1000 * ha / n},
            "engine_module_self_ms_per_line": {"before": 1000 * eb / n, "after": 1000 * ea / n},
            "engine_cum_ms_per_line": {
                "before": {k: 1000 * v / n for k, v in sorted(cb.items(), key=lambda kv: -kv[1])[:6]},
                "after": {k: 1000 * v / n for k, v in sorted(ca.items(), key=lambda kv: -kv[1])[:6]},
            },
        }
        report["modes"][mode] = entry
        print(f"\n[{mode}] outputs identical: {entry['identical']} "
              f"(max |diff| {max_diff:.3g}, same shapes {same_shape})")
        print(f"  wall ms/line  before {1000 * tb / n:.3f}  after {1000 * ta / n:.3f}"
              f"   ({n / tb:.1f} -> {n / ta:.1f} lines/s)")
        print(f"  host ms/line  before {1000 * hb / n:.3f}  after {1000 * ha / n:.3f}"
              f"   (all non-torch cProfile self time)")
        print(f"  engine-module self ms/line  before {1000 * eb / n:.3f}  after {1000 * ea / n:.3f}")
        for tag, cum in (("before", cb), ("after", ca)):
            top = sorted(cum.items(), key=lambda kv: -kv[1])[:5]
            print(f"  {tag} engine cum ms/line: " +
                  ", ".join(f"{k} {1000 * v / n:.3f}" for k, v in top))
    print("\n" + json.dumps({m: {"identical": v["identical"],
                                 "host_ms_per_line": v["host_ms_per_line"]}
                             for m, v in report["modes"].items()}))


if __name__ == "__main__":
    main()
