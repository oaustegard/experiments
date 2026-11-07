"""Build /tmp/claude-0/wd_launch.py: a Colab cell that writes model.py + train.py to /content/wd and starts
training detached. Usage: python3 run/colab_launch.py STEPS && colab exec -s wd -f /tmp/claude-0/wd_launch.py"""
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[1] / "train"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
cell = f'''
import os, subprocess, pathlib
d = pathlib.Path("/content/wd"); d.mkdir(exist_ok=True)
(d / "model.py").write_text({(here / "model.py").read_text()!r})
(d / "train.py").write_text({(here / "train.py").read_text()!r})
p = subprocess.Popen(f"cd /content/wd && nohup python train.py --out /content/wd/run --steps {steps} > /content/wd/stdout.txt 2>&1 &", shell=True)
print("launched")
'''
Path("/tmp/claude-0/wd_launch.py").write_text(cell)
Path("/tmp/claude-0/wd_peek.py").write_text('''
import subprocess, pathlib
print(subprocess.run("tail -c 1500 /content/wd/stdout.txt; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; ls /content/wd/run", shell=True, capture_output=True, text=True).stdout)
if pathlib.Path("/content/wd/run/DONE").exists(): print("DONE")
''')
print("ok")
