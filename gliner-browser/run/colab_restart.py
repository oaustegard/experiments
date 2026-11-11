# Run with `colab exec`: once A0 has reported, stop the running job and start the amended arm list
# (finished arms are skipped). The bracketed patterns keep pkill from matching its own shell.
import subprocess, time
from pathlib import Path
B = Path("/content/exp"); D = B / "gliner-browser"
def sh(c, **k): return subprocess.run(["bash", "-c", c], capture_output=True, text=True, **k)
t = time.time()
while not (D / "results" / "A0.json").exists() and time.time() - t < 240:
    time.sleep(5)
print("A0 result present:", (D / "results" / "A0.json").exists())
sh("pkill -f '[c]olab_job.py'; pkill -f 'run/[t]rain.py'; sleep 4")
print("left:", sh("pgrep -af '[c]olab_job|run/[t]rain.py'").stdout)
print(sh("git pull -q", cwd=B).stderr[-300:], sh("git log --oneline -1", cwd=B).stdout)
p = subprocess.Popen("python3 -u run/colab_job.py --arms A0,A9,A10,A11,A12 >> /content/driver_stdout.log 2>&1", shell=True,
                     cwd=D, start_new_session=True)
print("started pid", p.pid)
