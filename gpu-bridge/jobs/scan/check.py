"""Rebuild the scan input in numpy and check a scan result: python3 check.py result.json"""
import json, sys, time
import numpy as np
r = json.load(open(sys.argv[1])); o = r.get("outputs") or r.get("result")
n = o["n"]; i = np.arange(n, dtype=np.uint32)
with np.errstate(over="ignore"):
    h = i * np.uint32(0x9E3779B1); h ^= h >> np.uint32(16); h = h * np.uint32(0x85EBCA6B); h ^= h >> np.uint32(13)
v = (h & np.uint32(15)).astype(np.uint64)
t = []
for _ in range(5):
    t0 = time.perf_counter(); c = np.cumsum(v); t.append(time.perf_counter() - t0)
ex = np.concatenate([[0], c[:-1]]).astype(np.uint64) % 2**32
med = lambda a: sorted(a[1:])[len(a[1:]) // 2] if a else None
print(json.dumps({"device_mismatches": o["mismatches"], "checksum_ok": int(ex.sum() % 2**32) == o["checksum"],
  "samples_ok": bool((ex[o["sampleIdx"]] == np.array(o["samples"], dtype=np.uint64)).all()), "total_ok": int(c[-1] % 2**32) == o["total"],
  "gpu_median_ms": med(o.get("gpuMs") or []), "numpy_cumsum_ms": round(sorted(t)[2] * 1000, 2), "n": n}))
