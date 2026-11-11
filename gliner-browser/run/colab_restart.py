# Run with `colab exec`: stop the running job and start the amended arm list (finished arms are skipped, A0 resumes).
import subprocess, time
from pathlib import Path
B = Path("/content/exp")
def sh(c, **k): return subprocess.run(["bash", "-o", "pipefail", "-c", c], capture_output=True, text=True, **k)
print(sh("pkill -f colab_job.py; pkill -f run/train.py; sleep 5; pgrep -af 'colab_job|train.py' | grep -v pgrep").stdout)
print(sh("git pull -q", cwd=B).stderr[-300:], sh("git log --oneline -1", cwd=B).stdout)
p = subprocess.Popen("python3 -u run/colab_job.py --arms A0,A9,A10,A11,A12 >> /content/driver_stdout.log 2>&1", shell=True,
                     cwd=B / "gliner-browser", start_new_session=True)
print("started pid", p.pid)
