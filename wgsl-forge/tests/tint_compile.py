"""Compile every corpus shader, original and minified, with Chromium's WGSL compiler
(Tint, via createShaderModule + getCompilationInfo).

    python3 tests/tint_compile.py [--json results/tint_compile.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.gpu import run_jobs      # noqa: E402
from forge.minify import minify      # noqa: E402

FILES = sorted((HERE / "corpus").glob("*/*.wgsl"))


def main():
    jobs = []
    for f in FILES:
        src = f.read_text()
        jobs.append({"id": f"{f.name}|o", "kind": "compile", "wgsl": src})
        jobs.append({"id": f"{f.name}|m", "kind": "compile", "wgsl": minify(src)})
    res = run_jobs(jobs)
    counts = {"both": 0, "original_only": 0, "minified_only": 0, "neither": 0}
    cases = {}
    for f in FILES:
        o, m = res[f"{f.name}|o"]["ok"], res[f"{f.name}|m"]["ok"]
        k = "both" if o and m else "original_only" if o else "minified_only" if m else "neither"
        counts[k] += 1
        cases[f.name] = {"verdict": k, "original_error": res[f"{f.name}|o"].get("error"),
                         "minified_error": res[f"{f.name}|m"].get("error")}
    print(counts)
    for n, c in cases.items():
        if c["verdict"] != "both":
            print(f"{c['verdict']:<14} {n}: {(c['original_error'] or c['minified_error'] or '')[:120]}")
    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.write_text(json.dumps({"files": len(FILES), "counts": counts, "cases": cases}, indent=1))


if __name__ == "__main__":
    main()
