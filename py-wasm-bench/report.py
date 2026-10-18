"""results/*.json -> markdown tables: seconds, speed relative to native CPython 3.11
and to cpython-wasi, and shipped bytes. Usage: python3 report.py results/a.json [...]"""
import json
import sys
from collections import defaultdict

rows, label_of = [], {}
for path in sys.argv[1:]:
    data = json.load(open(path))
    tag = data.get("runtime", "")
    for r in data["rows"]:
        r = dict(r)
        if tag and not r["arm"].startswith("cpython-native") and not r["arm"].startswith("c-native"):
            r["arm"] = f'{r["arm"]} [{tag}]'
        rows.append(r)

kernels = ["nbody", "mandelbrot", "matmul", "dictwords"]
t = defaultdict(dict)
size = defaultdict(dict)
err = defaultdict(dict)
skipped = {}
for r in rows:
    if "skipped" in r:
        skipped[r["arm"]] = r["skipped"]
        continue
    if r.get("verified"):
        t[r["arm"]][r["kernel"]] = r["seconds"]
        size[r["arm"]][r["kernel"]] = r.get("bytes")
    else:
        err[r["arm"]][r["kernel"]] = r.get("error", "?")

base = t.get("cpython-native-3.11", {})
wasi = {k: v for a, d in t.items() if a.startswith("cpython-wasi") and "[wasmtime]" in a for k, v in d.items()}


def cell(a, k):
    if k in t[a]:
        s = t[a][k]
        x = f" ({base[k] / s:.2f}x)" if k in base else ""
        return f"{s:.3f}s{x}"
    if k in err[a]:
        e = err[a][k]
        return "unsupported" if "unsupported" in e else "FAIL"
    return "–"


arms = sorted(set(t) | set(err), key=lambda a: -sum(base.get(k, 0) / t[a][k] for k in t[a]) / max(1, len(t[a])))
print("Seconds inside the program around `run(n)`, median of reps; (Nx) = speed vs native CPython 3.11.\n")
print("| arm | " + " | ".join(kernels) + " |")
print("|---|" + "---|" * len(kernels))
for a in arms:
    print(f"| {a} | " + " | ".join(cell(a, k) for k in kernels) + " |")
if wasi:
    print("\nSpeed vs cpython-wasi under wasmtime (the in-wasm interpreter baseline):\n")
    print("| arm | " + " | ".join(kernels) + " |")
    print("|---|" + "---|" * len(kernels))
    for a in arms:
        print(f"| {a} | " + " | ".join(f"{wasi[k] / t[a][k]:.2f}x" if k in t[a] and k in wasi else "–" for k in kernels) + " |")
print("\nShipped bytes per kernel (artifact.txt):\n")
print("| arm | " + " | ".join(kernels) + " |")
print("|---|" + "---|" * len(kernels))
for a in arms:
    if any(size[a].get(k) for k in kernels):
        print(f"| {a} | " + " | ".join(f"{size[a][k] / 1e6:.2f} MB" if size[a].get(k) else "–" for k in kernels) + " |")
for a, why in skipped.items():
    print(f"\n`{a}` skipped: {why}")
for a, d in err.items():
    for k, e in d.items():
        print(f"\n`{a}` / `{k}`: {e[:200]}")
