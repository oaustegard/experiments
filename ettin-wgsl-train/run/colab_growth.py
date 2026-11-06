"""Run run/growth_python.py inside a Colab GPU kernel:

    colab new -s ewt --gpu L4
    colab exec -s ewt -f ettin-wgsl-train/run/colab_growth.py     # blocks for the whole run
    colab download -s ewt /content/ewt_growth.zip ettin-wgsl-train/fixtures/ewt_growth.zip
    colab stop -s ewt

Clones experiments at main (sparse), installs transformers 4.57.1, and zips the arm results.
Progress goes to /content/ewt.log.
"""
import subprocess
import zipfile
from pathlib import Path

BASE = Path("/content/exp")
D = BASE / "ettin-wgsl-train"
OUT = Path("/content/growth")
LOG = open("/content/ewt.log", "a")


def sh(cmd, **kw):
    LOG.write(f"$ {cmd}\n")
    LOG.flush()
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, **kw)


if not BASE.exists():
    sh(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments {BASE}")
    sh("git sparse-checkout set adapter-library/run adapter-library/results browser-decider/run browser-decider/data _lib ettin-wgsl-train/run", cwd=BASE)
sh("git pull -q", cwd=BASE)
sh("pip install -q transformers==4.57.1")
r = sh(f"python3 -I run/growth_python.py {OUT} 0-9 64,300 2>&1 | grep --line-buffered -v -i warning", cwd=D)
LOG.write(f"GROWTH_RC {r.returncode}\n")
with zipfile.ZipFile("/content/ewt_growth.zip", "w") as z:
    for f in OUT.glob("*.json"):
        z.write(f, f"growth/{f.name}")
    for f in (BASE / "adapter-library" / "out" / "adapters").glob("S-lora-*.pt"):
        z.write(f, f"adapters/{f.name}")
LOG.write("ALL_DONE\n")
LOG.flush()
print("done", r.returncode)
