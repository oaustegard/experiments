"""Build the Colab cells for a training run: /tmp/claude-0/wid_launch.py writes model.py + train.py to
/content/wid and starts training detached; /tmp/claude-0/wid_peek.py prints the log tail and the run dir.

    python3 run/colab_launch.py STEPS && colab exec -s wid --timeout 120 -f /tmp/claude-0/wid_launch.py
"""
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[1] / "train"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
Path("/tmp/claude-0").mkdir(exist_ok=True)
cell = f'''
import os, subprocess, pathlib
d = pathlib.Path("/content/wid"); d.mkdir(exist_ok=True)
(d / "model.py").write_text({(here / "model.py").read_text()!r})
(d / "train.py").write_text({(here / "train.py").read_text()!r})
subprocess.Popen(f"cd /content/wid && nohup python -u train.py --out /content/wid/run --data /content/wid/afhq.npz --steps {steps} > /content/wid/stdout.txt 2>&1 &", shell=True)
print("launched")
'''
Path("/tmp/claude-0/wid_launch.py").write_text(cell)
Path("/tmp/claude-0/wid_peek.py").write_text('''
import subprocess, pathlib
print(subprocess.run("tail -c 1200 /content/wid/stdout.txt; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; ls /content/wid/run; pgrep -f train.py >/dev/null && echo ALIVE || echo NOPROC", shell=True, capture_output=True, text=True).stdout)
if pathlib.Path("/content/wid/run/DONE").exists(): print("DONE")
''')
print("ok")
