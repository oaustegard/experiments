"""Write a self-contained Kaggle script that runs data.py then sweep.py for one model.

usage: make_kaggle.py MODEL N SETS [SCHEMES_FILE] > job.py
The three run/ modules are embedded so the kernel needs only pip packages and internet.
"""
import base64
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
model, n, sets = sys.argv[1], sys.argv[2], sys.argv[3]
schemes = Path(sys.argv[4] if len(sys.argv) > 4 else here / "schemes.txt").read_text().split()
files = {f: base64.b64encode((here / f).read_bytes()).decode() for f in ("quant.py", "data.py", "sweep.py")}
print(f'''import base64, subprocess, sys, os
from pathlib import Path
FILES = {files!r}
run = Path("/kaggle/working/run"); run.mkdir(parents=True, exist_ok=True)
for name, b in FILES.items():
    (run / name).write_bytes(base64.b64decode(b))
def sh(*a):
    print("+", *a, flush=True)
    p = subprocess.Popen(a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "arn" not in line: print(line, end="", flush=True)
    if p.wait(): sys.exit(p.returncode)
sh(sys.executable, "-m", "pip", "install", "-q", "jiwer", "soundfile")
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
sh(sys.executable, str(run / "data.py"))
sh(sys.executable, str(run / "sweep.py"), "--model", {model!r}, "--schemes", {";".join(schemes)!r},
   "--sets", {sets!r}, "--n", {n!r}, "--batch", "16", "--device", "cuda")
import shutil; shutil.rmtree("/kaggle/working/cache", ignore_errors=True); shutil.rmtree(run, ignore_errors=True)
print("JOB_DONE", flush=True)
''')
