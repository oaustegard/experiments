"""Minifier size benchmark on the corpus (86 shaders: naga's WGSL test inputs and the
webgpu-samples shaders).

Reports raw and gzip-9 bytes for: the original; whitespace/comment removal only (parse
and print with every pass off); the full minifier; the full minifier without type
aliases; each pass switched off in turn; and LucentFlux wgsl-minifier 0.7.0 (naga-based)
on the files it accepts. Every output is validated with naga.

    python3 tests/bench_minify.py [--json results/minify_bench.json]
"""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.minify import Options, minify   # noqa: E402

FILES = sorted((HERE / "corpus").glob("*/*.wgsl"))
PASSES = [f.name for f in fields(Options) if f.type in ("bool", bool)]


def gz(s):
    return len(gzip.compress(s.encode(), 9))


def naga_ok(src):
    with tempfile.NamedTemporaryFile("w", suffix=".wgsl", delete=False) as fh:
        fh.write(src)
    r = subprocess.run(["naga", fh.name], capture_output=True)
    Path(fh.name).unlink()
    return r.returncode == 0


def run(opts):
    raw = g = 0
    invalid = []
    for f in FILES:
        out = minify(f.read_text(), opts)
        raw += len(out)
        g += gz(out)
        if not naga_ok(out):
            invalid.append(f.name)
    return {"raw": raw, "gz": g, "invalid": invalid}


def baseline():
    exe = shutil.which("wgsl-minifier")
    if not exe:
        return None
    rows = []
    for f in FILES:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.wgsl"
            subprocess.run([exe, "-f", str(f), str(out)], capture_output=True)
            src = f.read_text()
            ours = minify(src)
            if out.exists():
                b = out.read_text()
                rows.append({"file": f.name, "orig": len(src), "ours": len(ours), "baseline": len(b),
                             "orig_gz": gz(src), "ours_gz": gz(ours), "baseline_gz": gz(b),
                             "baseline_valid": naga_ok(b)})
            else:
                rows.append({"file": f.name, "baseline": None})
    ok = [r for r in rows if r.get("baseline") is not None and r["baseline_valid"]]
    S = lambda k: sum(r[k] for r in ok)  # noqa: E731
    return {"accepted": len([r for r in rows if r.get("baseline") is not None]),
            "rejected_or_crashed": len([r for r in rows if r.get("baseline") is None]),
            "invalid_output": len([r for r in rows if r.get("baseline") is not None and not r["baseline_valid"]]),
            "compared": len(ok),
            "orig": S("orig"), "ours": S("ours"), "baseline": S("baseline"),
            "orig_gz": S("orig_gz"), "ours_gz": S("ours_gz"), "baseline_gz": S("baseline_gz"),
            "ours_ratio": round(S("ours") / S("orig"), 4), "baseline_ratio": round(S("baseline") / S("orig"), 4),
            "ours_gz_ratio": round(S("ours_gz") / S("orig_gz"), 4),
            "baseline_gz_ratio": round(S("baseline_gz") / S("orig_gz"), 4),
            "ours_smaller": sum(r["ours"] < r["baseline"] for r in ok)}


def main():
    orig_raw = sum(len(f.read_text()) for f in FILES)
    orig_gz = sum(gz(f.read_text()) for f in FILES)
    t0 = time.time()
    full = run(Options())
    full_s = time.time() - t0
    none = run(Options(**{p: False for p in PASSES}))
    res = {"files": len(FILES), "orig": {"raw": orig_raw, "gz": orig_gz},
           "whitespace_only": none, "full": full, "full_seconds": round(full_s, 2),
           "no_aliases": run(Options(type_aliases=False)), "ablation": {}}
    for p in PASSES:
        r = run(Options(**{p: False}))
        res["ablation"][p] = {"raw": r["raw"], "gz": r["gz"], "raw_cost": r["raw"] - full["raw"],
                              "gz_cost": r["gz"] - full["gz"], "invalid": r["invalid"]}
    res["ratios"] = {k: {"raw": round(res[k]["raw"] / orig_raw, 4), "gz": round(res[k]["gz"] / orig_gz, 4)}
                     for k in ("whitespace_only", "full", "no_aliases")}
    res["baseline"] = baseline()
    print(json.dumps({k: v for k, v in res.items() if k != "ablation"}, indent=1))
    for p, v in res["ablation"].items():
        print(f"without {p:<12} +{v['raw_cost']:6d} raw  +{v['gz_cost']:5d} gz  invalid {len(v['invalid'])}")
    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
