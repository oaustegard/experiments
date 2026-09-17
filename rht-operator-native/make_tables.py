"""Markdown tables for RESULTS.md, straight from ci/*.json."""
import json
from pathlib import Path

CI = Path(__file__).parent / "ci"
MACHINES = [
    ("local-xeon-avx512-1cpu", "Xeon AVX-512, 1 vCPU (authoring container)"),
    ("gha-ubuntu24-x64", "GitHub ubuntu-24.04 x64"),
    ("gha-ubuntu24-arm64", "GitHub ubuntu-24.04-arm (Neoverse V2)"),
    ("gha-macos15-arm64", "GitHub macos-15 (Apple Silicon, Accelerate)"),
]
DIMS = (384, 768, 1024, 1536, 2048, 3072, 4096)


def load():
    return {k: json.load(open(CI / f"{k}.json")) for k, _ in MACHINES if (CI / f"{k}.json").exists()}


def ratio(r, threads, d, shape):
    for s in r["speed"]:
        if s["threads"] == threads and s["d"] == d and s["shape"] == shape:
            return s["op_s"] / s["dense_s"]
    return None


def speed_table(runs, shape, all_threads):
    head = "| machine | threads | " + " | ".join(f"d={d}" for d in DIMS) + " |"
    out = [head, "|" + "---|" * (len(DIMS) + 2)]
    for k, name in MACHINES:
        if k not in runs:
            continue
        r = runs[k]
        t = r["ncpu"] if all_threads else 1
        if all_threads and t == 1:
            continue
        cells = [ratio(r, t, d, shape) for d in DIMS]
        out.append(f"| {name} | {t} | " + " | ".join("—" if c is None else f"{c:.2f}" for c in cells) + " |")
    return "\n".join(out)


def fingerprint_table(runs, d, keys):
    out = ["| field | " + " | ".join(k for k, _ in MACHINES if k in runs) + " | agree |",
           "|" + "---|" * (len([k for k, _ in MACHINES if k in runs]) + 2)]
    for key in keys:
        vals = [next(f for f in runs[k]["fingerprints"] if f["d"] == d).get(key, "—") for k, _ in MACHINES if k in runs]
        out.append(f"| `{key}` | " + " | ".join(f"`{v[:8]}`" for v in vals) + f" | {'**yes**' if len(set(vals)) == 1 else 'no (' + str(len(set(vals))) + ' distinct)'} |")
    return "\n".join(out)


if __name__ == "__main__":
    runs = load()
    print("## batch encode, n=10,000, 1 thread\n"); print(speed_table(runs, "enc10000", False))
    print("\n## batch encode, n=10,000, all threads\n"); print(speed_table(runs, "enc10000", True))
    print("\n## single query, 1 thread\n"); print(speed_table(runs, "query1d", False))
    print("\n## single query, all threads\n"); print(speed_table(runs, "query1d", True))
    for n in (64, 128, 256, 1024):
        print(f"\n## batch {n}, all threads\n"); print(speed_table(runs, f"enc{n}", True))
    keys = ["R", "op_rot", "dense_rot"] + [f"{p}{b}{s}" for b in (2, 4, 8) for p, s in
            (("cents", ""), ("bounds", ""), ("bounds", "_f32mid"), ("dense_codes", ""), ("op_codes", ""), ("op_codes", "_f32mid"))]
    for d in (768, 3072):
        print(f"\n## fingerprints d={d}\n"); print(fingerprint_table(runs, d, keys))
