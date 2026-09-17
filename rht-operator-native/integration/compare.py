"""Join bench.py outputs: main vs branch per machine, and across machines.

python compare.py <dir>   (a ci/<sha>/ directory, or out/)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
runs = {}
for p in sorted(D.glob("*__*.json")):
    label, variant = p.stem.split("__")
    runs[(label, variant)] = json.load(open(p))
labels = sorted({l for l, _ in runs})
print("# machines:", labels)
for l in labels:
    b = runs.get((l, "branch"), {})
    print(f"  {l}: native={b.get('native')} reason={b.get('native_reason')} blas={[x.get('architecture') for x in b.get('blas', [])][:1]}")

print("\n# fingerprints: distinct values across machines (1 = machine-independent)")
keys = sorted(next(iter(runs.values()))["fingerprints"])
print(f"{'key':<16} {'what':<6} {'main':>5} {'branch':>6}")
for k in keys:
    for what in ("bounds", "codes"):
        m = {runs[(l, 'main')]['fingerprints'][k][what] for l in labels if (l, 'main') in runs}
        b = {runs[(l, 'branch')]['fingerprints'][k][what] for l in labels if (l, 'branch') in runs}
        print(f"{k:<16} {what:<6} {len(m):>5} {len(b):>6}")

print("\n# share of codes that differ, main vs branch (and main vs main2 as a control)")
codes = D / "codes"
for l in labels:
    rows = []
    for k in keys:
        f = k.replace("/", "_")
        a, b, c = (codes / f"{l}__{v}__{f}.npy" for v in ("main", "branch", "main2"))
        if a.exists() and b.exists():
            A, B = np.load(a), np.load(b)
            ctl = (A != np.load(c)).mean() if c.exists() else float("nan")
            rows.append(f"{k}:{(A != B).mean():.1e}(ctl {ctl:.0e})")
    print(f"  {l}: " + "  ".join(rows))

print("\n# speed: branch / main (main2 / main is the noise control)")
metrics = [m for m in runs[(labels[0], "main")]["speed"][0] if m.endswith("_s")]
for l in labels:
    if (l, "branch") not in runs:
        continue
    print(f"## {l}")
    print(f"{'metric':<18}" + "".join(f"{'d=' + str(r['d']):>16}" for r in runs[(l, 'main')]["speed"]))
    for m in metrics:
        cells = []
        ctl = runs.get((l, "main2"), {}).get("speed") or [None] * len(runs[(l, "main")]["speed"])
        for rm, rb, r2 in zip(runs[(l, "main")]["speed"], runs[(l, "branch")]["speed"], ctl):
            c = f"{r2[m] / rm[m]:4.2f}" if r2 else " n/a"
            cells.append(f"{rb[m] / rm[m]:6.2f} ({c})")
        print(f"{m:<18}" + "".join(f"{c:>16}" for c in cells))
    print("  main absolute build_s:", [round(r["build_s"], 4) for r in runs[(l, "main")]["speed"]],
          " branch:", [round(r["build_s"], 5) for r in runs[(l, "branch")]["speed"]])

print("\n# cutoff sweep (branch): best cutoff per (d, n), and seconds at each")
for l in labels:
    sw = runs.get((l, "branch"), {}).get("sweep")
    if not sw:
        continue
    t = defaultdict(dict)
    for r in sw:
        t[(r["d"], r["n"])][r["cutoff"]] = r["s"]
    print(f"## {l}")
    for (d, n), m in sorted(t.items()):
        best = min(m, key=m.get)
        print(f"  d={d} n={n:<4} best=2^{best.bit_length() - 1:<3} " + " ".join(f"2^{c.bit_length() - 1}:{s * 1e6:7.1f}us" for c, s in sorted(m.items())))

for p in sorted(D.glob("*__pytest.txt")):
    print("\n#", p.name, "::", p.read_text().strip().splitlines()[-1] if p.read_text().strip() else "(empty)")
