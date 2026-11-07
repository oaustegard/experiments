"""Build the Colab cells for a training run named NAME (VM dir /content/NAME, Colab session NAME):
/tmp/claude-0/NAME_launch.py writes model.py + train.py and starts training detached;
/tmp/claude-0/NAME_peek.py prints the log tail, the run dir and whether the trainer is alive.

    python3 run/colab_launch.py STEPS [NAME=wid] [DATASET=afhq]
    colab exec -s NAME --timeout 120 -f /tmp/claude-0/NAME_launch.py
"""
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[1] / "train"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
name = sys.argv[2] if len(sys.argv) > 2 else "wid"
dataset = sys.argv[3] if len(sys.argv) > 3 else "afhq"
data = "afhq.npz" if dataset == "afhq" else f"{dataset}.npz"
Path("/tmp/claude-0").mkdir(exist_ok=True)
cell = f'''
import os, subprocess, pathlib
d = pathlib.Path("/content/{name}"); d.mkdir(exist_ok=True)
(d / "model.py").write_text({(here / "model.py").read_text()!r})
(d / "train.py").write_text({(here / "train.py").read_text()!r})
subprocess.Popen("cd /content/{name} && nohup python3 -u train.py --out /content/{name}/run --data /content/{name}/{data} --dataset {dataset} --steps {steps} > /content/{name}/stdout.txt 2>&1 &", shell=True)
print("launched")
'''
Path(f"/tmp/claude-0/{name}_launch.py").write_text(cell)
Path(f"/tmp/claude-0/{name}_peek.py").write_text(f'''
import subprocess, pathlib
print(subprocess.run("tail -c 1200 /content/{name}/stdout.txt; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; ls /content/{name}/run; pgrep -f 'train.py --out /content/{name}/' >/dev/null && echo ALIVE || echo NOPROC", shell=True, capture_output=True, text=True).stdout)
if pathlib.Path("/content/{name}/run/DONE").exists(): print("DONE")
''')
print("ok")
