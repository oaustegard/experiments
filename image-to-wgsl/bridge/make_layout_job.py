"""Spec for the payload-layout comparison: one array<u32>, one array<vec4u>, or split u32 arrays."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import codec  # noqa: E402

SIZES = [[768, 512], [1920, 1280]]
VARIANTS = [("results/runs/kodim23_prefix1024.json", "u32", None),
            ("results/runs/kodim23_prefix1024.json", "vec4u", None),
            ("results/runs/kodim23_prefix1024.json", "split", 512),
            ("scratch/synth_4096.json", "vec4u", None),
            ("scratch/synth_4096.json", "split", 2047)]

spec = []
for path, store, chunk in VARIANTS:
    run, _ = codec.load_run(HERE / path)
    r = codec.rate(codec.emit_wgsl(run, store=store, chunk=chunk))
    n = int(run["P"].shape[0])
    sid = f"{Path(path).stem}_n{n}_{store}" + (f"{chunk}" if chunk else "")
    spec.append({"id": sid, "wgsl": r["wgsl"], "w": run["W"], "h": run["H"], "sizes": SIZES, "reps": 5,
                 "run": path, "n": n, "store": store, "chunk": chunk, "gz": r["gz"], "min": r["min"]})
    print(sid, "gz", r["gz"], "min", r["min"])
Path(sys.argv[1]).write_text(json.dumps(spec))
