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
# speed claims quoted in RESULTS.md (op_s / dense_s), last run
rat = lambda x: x["op_s"] / x["dense_s"]
big = [rat(x) for r in runs.values() for x in r["speed"] if x["d"] >= 1024]
check(round(max(big), 2) <= 0.48 and round(min(big), 3) >= 0.007, f"d>=1024: every cell in [0.007, 0.48] (got {min(big):.3f}..{max(big):.3f})")
x64 = runs["gha-ubuntu24-x64"]
check(0.9 <= ratio(x64, 4, 768, "enc128") <= 1.1 and 0.9 <= ratio(x64, 4, 768, "enc256") <= 1.1, "x64 d=768 n=128/256 at parity")
multi = [r for r in runs.values() if r["ncpu"] > 1]
check(all(1.3 <= ratio(r, r["ncpu"], 384, f"enc{n}") <= 1.85 for r in multi for n in (64, 128, 256)), "d=384 n=64..256: dense wins 1.3-1.85 on every multi-core machine")
arm = runs["gha-ubuntu24-arm64"]
check(abs(ratio(arm, 4, 384, "enc10000") - 0.37) < 0.01, "ARM d=384 batch 0.37")
check(1.35 <= ratio(arm, 4, 384, "enc256") <= 1.45 and ratio(arm, 4, 384, "enc1024") < 0.4, "ARM d=384: n=256 serial ~1.40, n=1024 parallel < 0.4")
check(ratio(runs["local-xeon-avx512-1cpu"], 1, 384, "enc10000") > 1.0, "d=384: dense wins batch on the AVX-512 box")
check(1.1 <= ratio(x64, 1, 384, "enc10000") <= 1.3, "x64 (SkylakeX this run) d=384 1-thread batch in 1.1-1.3")
m = runs["gha-macos15-arm64"]
check(0.6 < ratio(m, m["ncpu"], 768, "enc10000") < 0.9, "macOS d=768 batch in (0.6, 0.9)")
check(all(0.15 <= ratio(r, 1, 768, "query1d") <= 0.40 for r in runs.values()), "d=768 single query 0.15-0.40 one thread")
sys.exit(1 if fails else 0)
