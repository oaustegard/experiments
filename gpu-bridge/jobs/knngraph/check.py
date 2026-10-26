"""Rebuild the points and check a knngraph result: python3 check.py result.json [--sample S]"""
import json, sys, time
import numpy as np
r = json.load(open(sys.argv[1])); o = r.get("outputs") or r.get("result")
n, D, K = o["n"], 256, 10
i = np.arange(n * D, dtype=np.uint64) + 777
i = i.astype(np.uint32)
with np.errstate(over="ignore"):
    h = i * np.uint32(0x9E3779B1); h ^= h >> np.uint32(16); h = h * np.uint32(0x85EBCA6B); h ^= h >> np.uint32(13)
    h = h * np.uint32(0xC2B2AE35); h ^= h >> np.uint32(16)
pts = ((h & np.uint32(0xFFFFFF)).astype(np.float32) / np.float32(16777216) - np.float32(0.5)).reshape(n, D)
if "npy" in o["idx"]:
    got = np.load(o["idx"]["npy"]).reshape(n, K)
else:
    import base64; got = np.frombuffer(base64.b64decode(o["idx"]["b64"]), dtype=np.uint32).reshape(n, K)
t0 = time.perf_counter(); ref = np.empty((n, K), dtype=np.int64); B = 2048
for s in range(0, n, B):
    sc = pts[s:s + B] @ pts.T
    sc[np.arange(sc.shape[0]), np.arange(s, s + sc.shape[0])] = -np.inf
    part = np.argpartition(-sc, K, axis=1)[:, :K]
    ref[s:s + B] = np.take_along_axis(part, np.argsort(-np.take_along_axis(sc, part, 1), 1), 1)
cpu_s = time.perf_counter() - t0
recall = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(got, ref)]))
print(json.dumps({"n": n, "exact_order_match": float((got == ref).mean()), "recall@10": recall, "rows_fully_correct": float(np.mean([set(a) == set(b) for a, b in zip(got, ref)])),
  "gpu_total_s": (o.get("gpuTotalMs") or 0) / 1000, "phone_wall_s": o["totalWallMs"] / 1000, "numpy_s": round(cpu_s, 2), "gflop": o["gflop"]}))
