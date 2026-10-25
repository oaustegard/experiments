"""Rebuild the sort keys in numpy and check a sort result: python3 check.py result.json"""
import json, sys, time
import numpy as np
r = json.load(open(sys.argv[1])); o = r.get("outputs") or r.get("result")
n = o["n"]; i = np.arange(n, dtype=np.uint32) + np.uint32(12345)
with np.errstate(over="ignore"):
    h = i * np.uint32(0x9E3779B1); h ^= h >> np.uint32(16); h = h * np.uint32(0x85EBCA6B); h ^= h >> np.uint32(13)
    h = h * np.uint32(0xC2B2AE35); h ^= h >> np.uint32(16)
t = []
for _ in range(5):
    t0 = time.perf_counter(); s = np.sort(h); t.append(time.perf_counter() - t0)
w = (np.arange(n) % 1000 + 1).astype(np.uint64)
med = lambda a: sorted(a[1:])[len(a[1:]) // 2] if a else None
print(json.dumps({"device_mismatches": o["mismatches"], "checksum_ok": int((s.astype(np.uint64) * w).sum() % 2**32) == o["checksum"],
  "samples_ok": bool((s[o["sampleIdx"]] == np.array(o["samples"], dtype=np.uint32)).all()),
  "gpu_median_ms": med(o.get("gpuMs") or []), "numpy_sort_ms": round(sorted(t)[2] * 1000, 2), "n": n}))
