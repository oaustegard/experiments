"""Run every GPU arm of run/lora.py inside a Colab kernel:

    colab new -s al --gpu L4
    colab exec -s al -f run/colab_job.py      # blocks for the whole run: keeps the VM from idling out
    colab download -s al /content/al_out.zip out/al_out.zip

Clones adapter-library, browser-decider (the trainer it imports) and _lib at main, installs
transformers 4.57.1, runs lora.py (resume-safe), and zips results/ plus out/probs and out/adapters.
Progress goes to /content/al.log; a crash behind a pipe is not reported as rc=0 (pipefail).
"""
import subprocess
import zipfile
from pathlib import Path

BASE = Path("/content/exp")
D = BASE / "adapter-library"
LOG = open("/content/al.log", "a")


def sh(cmd, **kw):
    LOG.write(f"$ {cmd}\n")
    LOG.flush()
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, **kw)


if not D.exists():
    sh(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments {BASE}")
    sh("git sparse-checkout set adapter-library browser-decider/run browser-decider/data _lib", cwd=BASE)
sh("git pull -q", cwd=BASE)
sh("pip install -q transformers==4.57.1")
r = sh("python3 -I run/lora.py 2>&1 | grep -v -i warning", cwd=D)
LOG.write(f"LORA_RC {r.returncode}\n")
with zipfile.ZipFile("/content/al_out.zip", "w") as z:
    for f in (D / "results").glob("*.json"):
        z.write(f, f"results/{f.name}")
    for sub in ("probs", "adapters"):
        for f in (D / "out" / sub).glob("*"):
            z.write(f, f"out/{sub}/{f.name}")
LOG.write("ALL_DONE\n")
LOG.flush()
print("done", r.returncode)
