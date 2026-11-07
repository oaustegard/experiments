"""Build /tmp/claude-0/wid_eval.py: a Colab cell that installs clean-fid, writes eval.py next to the
trainer in /content/wid and starts the evaluation detached (log: /content/wid/eval_stdout.txt, ends
with 'eval done'). Usage: python3 run/colab_eval.py && colab exec -s wid --timeout 300 -f /tmp/claude-0/wid_eval.py
"""
from pathlib import Path

here = Path(__file__).resolve().parents[1] / "train"
cell = f'''
import subprocess, pathlib
d = pathlib.Path("/content/wid")
for name, src in {{"model.py": {(here / "model.py").read_text()!r}, "train.py": {(here / "train.py").read_text()!r},
                  "eval.py": {(here / "eval.py").read_text()!r}}}.items():
    (d / name).write_text(src)
print(subprocess.run("pip install -q clean-fid", shell=True, capture_output=True, text=True).returncode)
subprocess.Popen("cd /content/wid && nohup python3 -u eval.py --ema /content/wid/run/ema.pt --data /content/wid/afhq.npz --out /content/wid/eval > /content/wid/eval_stdout.txt 2>&1 &", shell=True)
print("launched")
'''
Path("/tmp/claude-0/wid_eval.py").write_text(cell)
print("ok")
