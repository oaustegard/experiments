"""Keep the writeup and the artifacts from drifting apart, in under 5 minutes.

    python3 recheck.py          # unit tests, minifier ratios, evidence refs (CPU, ~1 min)
    python3 recheck.py --gpu    # plus the compute, SVG and Python GPU suites (~1 more min)
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from forge.minify import minify   # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


r = subprocess.run([sys.executable, str(HERE / "tests" / "test_forge.py")], capture_output=True, text=True)
check("unit tests", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])

bench = json.loads((HERE / "results" / "minify_bench.json").read_text())
files = sorted((HERE / "corpus").glob("*/*.wgsl"))
raw = sum(len(minify(f.read_text())) for f in files)
gz = sum(len(gzip.compress(minify(f.read_text()).encode(), 9)) for f in files)
check("corpus size", len(files) == bench["files"], f"{len(files)} files")
check("minified raw bytes match results/minify_bench.json", raw == bench["full"]["raw"],
      f"{raw} vs {bench['full']['raw']}")
check("minified gzip bytes match", gz == bench["full"]["gz"], f"{gz} vs {bench['full']['gz']}")

r = subprocess.run([sys.executable, str(ROOT / "_lib" / "evidence.py"), str(HERE / "RESULTS.md")],
                   capture_output=True, text=True)
check("evidence refs", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else "")
r = subprocess.run([sys.executable, str(ROOT / "_lib" / "results_lint.py"), str(HERE / "RESULTS.md")],
                   capture_output=True, text=True)
check("results lint", r.returncode == 0, r.stdout.strip())

if "--gpu" in sys.argv:
    r = subprocess.run([sys.executable, str(HERE / "tests" / "diff_compute.py")], capture_output=True, text=True)
    want = json.loads((HERE / "results" / "diff_compute.json").read_text())["counts"]
    check("compute differential", str(want) in r.stdout, r.stdout.strip().splitlines()[-1])
    r = subprocess.run([sys.executable, str(HERE / "tests" / "diff_svg.py")], capture_output=True, text=True)
    want = json.loads((HERE / "results" / "diff_svg.json").read_text())["summary"]
    got = r.stdout.strip().splitlines()[-1]
    check("svg median psnr", f"'median_psnr': {want['median_psnr']}" in got and "'all_minified_identical': True" in got, got)

sys.exit(1 if fails else 0)
