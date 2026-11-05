"""Train the account decider's arms on a Colab GPU. Runs inside the Colab kernel:

    colab upload -s NAME data/cases.jsonl /content/cases.jsonl
    colab exec -s NAME -f run/colab_train.py

The kernel stays busy for the whole run (subprocess.run blocks), so Colab does not reclaim the VM as
idle, which happened to a detached run in browser-decider. Clones browser-decider (the trainer) and this
directory at main, trains every arm in ARMS with `train_enc.py --data`, and zips results/ plus the
checkpoints in KEEP to /content/bad_out.zip. Resume-safe: an arm whose results file exists is skipped.
"""
import subprocess
import zipfile
from pathlib import Path

ARMS = [(f"all-s{s}", ["--seed", str(s)]) for s in (0, 1, 2)]
ARMS += [(f"n{n}-s{s}", ["--seed", str(s), "--max-train-accounts", str(n)]) for n in (100, 300, 700) for s in (0, 1, 2)]
KEEP = ["all-s0"]
BASE = Path("/content/exp")
OUT = Path("/content/bad")
LOG = open("/content/bad.log", "a")


def sh(cmd, **kw):
    LOG.write(f"$ {cmd}\n")
    LOG.flush()
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, **kw)


if not (BASE / "browser-decider").exists():
    sh(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments {BASE}")
    sh("git sparse-checkout set browser-decider bsky-account-decider", cwd=BASE)
sh("git pull -q", cwd=BASE)
sh("pip install -q transformers==4.57.1")
OUT.mkdir(exist_ok=True)
for name, args in ARMS:
    if (OUT / "results" / f"{name}.json").exists():
        continue
    r = sh(f"python3 run/train_enc.py {name} --init stock --epochs 10 --data /content/cases.jsonl "
           f"--out-root {OUT} {' '.join(args)} 2>&1 | grep -v -i warning", cwd=BASE / "browser-decider")
    LOG.write(f"ARM_DONE {name} rc={r.returncode}\n")
    LOG.flush()
with zipfile.ZipFile("/content/bad_out.zip", "w") as z:
    for f in (OUT / "results").glob("*.json"):
        z.write(f, f"results/{f.name}")
    for name in KEEP:
        for f in (OUT / "out" / "ckpt" / name).glob("*"):
            z.write(f, f"out/ckpt/{name}/{f.name}")
LOG.write("ALL_ARMS_DONE\n")
LOG.flush()
print("done")
