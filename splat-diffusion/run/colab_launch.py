"""Build a Colab cell that writes the training code to /content/sd and runs fit -> train detached.

    python3 run/colab_launch.py K STEPS   ->  /tmp/claude-0/sd_launch.py, /tmp/claude-0/sd_peek.py
    colab exec -s sd -f /tmp/claude-0/sd_launch.py
"""
import sys
from pathlib import Path

src = Path(__file__).resolve().parents[1] / "train"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 512
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
files = {n: (src / n).read_text() for n in ("splats.py", "model.py", "fit_dataset.py", "train.py")}
cell = f'''
import pathlib, subprocess
d = pathlib.Path("/content/sd"); d.mkdir(exist_ok=True)
for n, t in {files!r}.items():
    (d / n).write_text(t)
subprocess.Popen("cd /content/sd && nohup sh -c 'python fit_dataset.py --out run --K {K} --bs 32 --steps 1000 && python train.py --out run --steps {steps}' > stdout.txt 2>&1 &", shell=True)
print("launched")
'''
Path("/tmp/claude-0/sd_launch.py").write_text(cell)
Path("/tmp/claude-0/sd_peek.py").write_text('''
import subprocess, pathlib
print(subprocess.run("tail -c 1200 /content/sd/stdout.txt; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; ls /content/sd/run", shell=True, capture_output=True, text=True).stdout)
if pathlib.Path("/content/sd/run/DONE").exists(): print("DONE")
''')
print("ok")
