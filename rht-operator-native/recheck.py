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
    check(len({f["dense_rot"] for f in fps.values()}) == len(fps), f"d={d}: dense rotation differs on every machine")
    for b in (2, 4, 8):
        check(same(f"cents{b}"), f"d={d}/{b}-bit: float32 centroids identical everywhere")
        check(same(f"bounds{b}_f32mid"), f"d={d}/{b}-bit: float32-midpoint boundaries identical everywhere")
        check(same(f"op_codes{b}_f32mid"), f"d={d}/{b}-bit: operator + f32-midpoint codes identical everywhere")
# speed claims quoted in RESULTS.md (op_s / dense_s, batch n=10,000)
for k in ("gha-ubuntu24-x64", "gha-ubuntu24-arm64"):
    r = runs[k]; t = r["ncpu"]
    check(all(ratio(r, t, d, "enc10000") <= 0.50 for d in (768, 1024, 1536, 2048, 3072, 4096)), f"{k}: <= 0.50 from d=768 up, all threads")
    check(all(ratio(r, t, d, "enc10000") <= 0.25 for d in (1024, 1536, 2048, 3072, 4096)), f"{k}: <= 0.25 from d=1024 up, all threads")
    check(all(ratio(r, t, d, "enc10000") <= 0.12 for d in (2048, 3072, 4096)), f"{k}: <= 0.12 from d=2048 up, all threads")
for k, r in runs.items():
    check(all(ratio(r, 1, d, "enc10000") <= 0.45 for d in (1024, 1536, 2048, 3072, 4096)), f"{k}: <= 0.45 from d=1024 up, 1 thread")
    check(ratio(r, 1, 4096, "query1d") <= 0.02, f"{k}: single query <= 0.02 at d=4096")
m = runs["gha-macos15-arm64"]
check(0.6 < ratio(m, m["ncpu"], 768, "enc10000") < 1.0, "macOS: serial kernel vs Accelerate is a narrow win at d=768")
check(ratio(runs["local-xeon-avx512-1cpu"], 1, 384, "enc10000") > 1.0, "d=384: dense still wins batch on the AVX-512 box")
sys.exit(1 if fails else 0)
