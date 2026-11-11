# Run with `colab exec` after the queue reports ALL_DONE: resume the arms that stopped on one OOM-skipped batch (finished arms are skipped).
import subprocess
from pathlib import Path
B = Path("/content/exp"); D = B / "gliner-browser"
def sh(c, **k): return subprocess.run(["bash", "-c", c], capture_output=True, text=True, **k)
print(sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader").stdout)
print("running:", sh("pgrep -af '[c]olab_job|run/[t]rain.py'").stdout)
print(sh("git pull -q", cwd=B).stderr[-300:], sh("git log --oneline -1", cwd=B).stdout)
p = subprocess.Popen("python3 -u run/colab_job.py --arms A9,A10,A11,A12 >> /content/driver_stdout.log 2>&1", shell=True, cwd=D, start_new_session=True)
print("started pid", p.pid)
