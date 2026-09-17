"""Check RESULTS.md's claims against ci/*.json. Sub-second; no compute."""
import json, sys
from pathlib import Path
from make_tables import load, ratio

runs = load()
fails = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

check(len(runs) == 4, f"four machines present ({sorted(runs)})")
for k, r in runs.items():
    check(all(e["all_equal"] for e in r["bitexact"]), f"{k}: operator bit-identical across builds, threads and NumPy")
for d in (768, 3072):
    fps = {k: next(f for f in r["fingerprints"] if f["d"] == d) for k, r in runs.items()}
    same = lambda key: len({f[key] for f in fps.values()}) == 1
    check(same("R"), f"d={d}: materialized R identical everywhere")
    check(same("op_rot"), f"d={d}: operator rotation identical everywhere")
    check(len({f["dense_rot"] for f in fps.values()}) >= 3, f"d={d}: dense rotation takes >= 3 distinct values across 4 machines")
    for b in (2, 4, 8):
        check(same(f"cents{b}"), f"d={d}/{b}-bit: float32 centroids identical everywhere")
        check(same(f"bounds{b}_f32mid"), f"d={d}/{b}-bit: float32-midpoint boundaries identical everywhere")
        check(same(f"op_codes{b}_f32mid"), f"d={d}/{b}-bit: operator + f32-midpoint codes identical everywhere")
# speed claims quoted in RESULTS.md: ranges across every archived run
from history import runs as hist_runs, cells, span
H = hist_runs()
local = json.load(open(Path(__file__).parent / "ci" / "history" / "local-xeon-avx512-1cpu.json"))
allr = [r for m in H.values() for r in m.values()]
by = lambda lab: [m[lab] for m in H.values()]
def within(vals, lo, hi, msg):
    a, b = span(vals)
    check(a is not None and round(a, 2) >= lo and round(b, 2) <= hi, f"{msg}: [{lo}, {hi}] (got {a:.3f}..{b:.3f})")
within((v for r in allr + [local] for d in (1024, 1536, 2048, 3072, 4096) for v in cells(r, d=d)), 0.0, 0.49, "d>=1024 every cell")
check(round(min(v for r in allr for d in (1024, 1536, 2048, 3072, 4096) for v in cells(r, d=d)), 3) >= 0.007, "d>=1024 floor 0.007")
within((v for r in allr for v in cells(r, "all", 768, "enc10000")), 0.19, 0.87, "d=768 batch 10k, all threads")
within((v for r in allr + [local] for v in cells(r, 1, 768, "query1d")), 0.17, 0.41, "d=768 single query, 1 thread")
within((v for r in by("gha-ubuntu24-x64") for n in (128, 256) for v in cells(r, "all", 768, f"enc{n}")), 0.82, 1.04, "x64 d=768 n=128/256")
within((v for r in by("gha-macos15-arm64") for t in (1, "all") for v in cells(r, t, 768, "enc10000")), 0.66, 0.87, "macOS d=768 batch")
within((v for r in allr if r["ncpu"] > 1 for n in (64, 128, 256) for v in cells(r, "all", 384, f"enc{n}")), 1.11, 1.81, "d=384 n=64..256 multi-core")
check(round(next(cells(local, 1, 384, "enc10000")), 2) == 1.20, "authoring box d=384 batch 1.20")
sky = [r for r in by("gha-ubuntu24-x64") if r["blas"][0]["architecture"] == "SkylakeX"]
has = [r for r in by("gha-ubuntu24-x64") if r["blas"][0]["architecture"] == "Haswell"]
check(len(sky) == 1 and len(has) == 3, "x64 kernels: 1 SkylakeX, 3 Haswell")
within((v for r in sky for t in (1, "all") for v in cells(r, t, 384, "enc10000")), 1.16, 1.29, "x64 SkylakeX d=384 batch")
within((v for r in has for v in cells(r, "all", 384, "enc10000")), 0.77, 0.93, "x64 Haswell d=384 batch, all threads")
within((v for r in by("gha-macos15-arm64") for t in (1, "all") for v in cells(r, t, 384, "enc10000")), 0.74, 1.22, "macOS d=384 batch")
within((v for r in by("gha-ubuntu24-arm64") for t in (1, "all") for v in cells(r, t, 384, "enc10000")), 0.36, 0.38, "ARM d=384 batch")
within((v for r in by("gha-ubuntu24-arm64") for v in cells(r, "all", 384, "enc256")), 1.30, 1.40, "ARM d=384 n=256")
within((v for r in by("gha-ubuntu24-arm64") for v in cells(r, "all", 384, "enc1024")), 0.36, 0.38, "ARM d=384 n=1024")
for key, lim in (("enc10000", 0.011), ("query1d", 0.11)):
    for d in (384, 768, 1024, 1536, 2048, 3072, 4096):
        a, b = span(v for r in by("gha-ubuntu24-arm64") for v in cells(r, "all", d, key))
        check(b - a <= lim, f"ARM run-to-run spread <= {lim} at d={d} {key} ({b - a:.3f})")
sys.exit(1 if fails else 0)
