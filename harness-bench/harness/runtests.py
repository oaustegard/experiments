#!/usr/bin/env python3
"""Hidden-suite runner the agent may call: `python3 harness/runtests.py <lang> <task>`.

Grades the agent's current solution file exactly the way the final grader does —
pristine tree, only the declared solution files overlaid — and prints the suite
output. The test sources live in a scratch tree the agent is never handed, so the
loop gets real failure evidence without putting the assertions in its context.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bench

lang, task = sys.argv[1], sys.argv[2]
arm = os.environ.get("ARM", "toolloop")
src = bench.ROOT / "work" / arm / lang / task
if not src.exists():
    sys.exit(f"no working dir for {arm}/{lang}/{task}")

tmp = Path(tempfile.mkdtemp(prefix="rt-"))
dst = tmp / task
shutil.copytree(bench.ex_dir(lang, task), dst)
for s in bench.solution_files(lang, task):
    if (src / s).exists():
        shutil.copy2(src / s, dst / s)

spec = bench.LANGS[lang]
env = dict(os.environ, GOFLAGS="-mod=mod", GOPATH="/tmp/gopath",
           CARGO_TARGET_DIR=str(dst / "_target"))
try:
    r = subprocess.run(spec["cmd"], cwd=dst, capture_output=True, text=True,
                       timeout=spec["timeout"], env=env)
    out, code = (r.stdout + r.stderr), r.returncode
except subprocess.TimeoutExpired:
    out, code = f"TIMEOUT after {spec['timeout']}s — the implementation is too slow", 1
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# strip the scratch path so the agent cannot walk back to the test sources
print(out.replace(str(tmp), "<hidden>")[-8000:])
print(f"\n=== {'PASS' if code == 0 else 'FAIL'} ===")
sys.exit(0)
