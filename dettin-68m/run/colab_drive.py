"""Start a detached job on a Colab session and stay with it until its log says ALL_DONE.

    python3 run/colab_drive.py SESSION run/colab_a.py --log /content/dettin_a.log \
        --pull /content/dettin_a.zip:out/colab/dettin_a.zip [--final REMOTE:LOCAL ...] [--attach]

The session must exist (`colab new -s SESSION --gpu L4`). The starter clones this repo at main
(sparse), then launches the script with Popen in its own process group and returns, because a long
`colab exec` can lose its connection. This process is then the tracked local job: every PERIOD seconds
it touches the kernel (an untouched VM is reclaimed after about 20 minutes), downloads the log and the
--pull files, and prints the newest progress line. It prints `done -> <log>` when the log ends with
ALL_DONE and exits 2 if the session stops answering. --attach skips the start and only follows.
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPARSE = "browser-decider/run browser-decider/data adapter-library/run _lib dettin-68m/run dettin-68m/data"
PERIOD = 240

STARTER = '''
import subprocess, os
from pathlib import Path
B = Path("/content/exp")
def sh(c, **k): return subprocess.run(["bash", "-o", "pipefail", "-c", c], capture_output=True, text=True, **k)
if not B.exists():
    print(sh("git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments /content/exp").stderr[-300:])
    print(sh("git sparse-checkout set {sparse}", cwd=B).stderr[-300:])
print(sh("git pull -q", cwd=B).stderr[-300:])
print(sh("git log --oneline -1", cwd=B).stdout)
p = subprocess.Popen("python3 -u {script} >> /content/driver_stdout.log 2>&1", shell=True, cwd=B / "dettin-68m",
                     start_new_session=True)
print("started pid", p.pid)
'''


def colab(*args, timeout=300):
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("script")
    ap.add_argument("--log", required=True)
    ap.add_argument("--pull", action="append", default=[])
    ap.add_argument("--final", action="append", default=[])
    ap.add_argument("--attach", action="store_true")
    a = ap.parse_args()
    local_log = ROOT / "out" / "colab" / Path(a.log).name
    local_log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        if not a.attach:
            st = Path(td) / "starter.py"
            st.write_text(STARTER.format(sparse=SPARSE, script=a.script))
            r = colab("exec", "-s", a.session, "-f", str(st), "--timeout", "600", timeout=700)
            print("start:", (r.stdout + r.stderr).strip()[-600:], flush=True)
        touch = Path(td) / "touch.py"
        touch.write_text("print('ok')\n")
        misses, last = 0, ""
        while True:
            time.sleep(PERIOD)
            colab("exec", "-s", a.session, "-f", str(touch), "--timeout", "60", timeout=90)
            r = colab("download", "-s", a.session, a.log, str(local_log))
            if r.returncode != 0 or not local_log.exists():
                misses += 1
                print(f"miss {misses}: {(r.stdout + r.stderr).strip()[-200:]}", flush=True)
                if misses >= 5:
                    print("session lost", flush=True)
                    sys.exit(2)
                continue
            misses = 0
            for spec in a.pull:
                remote, local = spec.split(":")
                (ROOT / local).parent.mkdir(parents=True, exist_ok=True)
                colab("download", "-s", a.session, remote, str(ROOT / local))
            lines = [l for l in local_log.read_text(errors="replace").splitlines() if l.strip()]
            marks = [l for l in lines if l.startswith(("ARM_DONE", "LR_CHOSEN", "QUANT_DONE", "PACK_DONE",
                                                         "SPIKES_DONE", "SPEED_DONE", "STAGE", "ALL_DONE"))]
            tail = lines[-1][:160] if lines else ""
            if tail != last:
                print(f"{time.strftime('%H:%M:%S')} marks={len(marks)} | {tail}", flush=True)
                last = tail
            if lines and lines[-1].strip() == "ALL_DONE":
                for spec in a.pull + a.final:
                    remote, local = spec.split(":")
                    (ROOT / local).parent.mkdir(parents=True, exist_ok=True)
                    r = colab("download", "-s", a.session, remote, str(ROOT / local), timeout=1800)
                    print(f"final {remote}: rc={r.returncode}", flush=True)
                print(f"done -> {local_log}", flush=True)
                return


if __name__ == "__main__":
    main()
