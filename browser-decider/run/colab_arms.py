"""Run training arms on a Colab GPU VM (executed there with `colab exec -s NAME -f run/colab_arms.py`).

Clones this repo's browser-decider directory at main, installs transformers, fetches the 10 MB build's
fp32 checkpoint from the ettin-native release, runs every arm in ARMS (resume-safe: an arm whose
results file exists is skipped), and zips results/ plus the checkpoints named in KEEP to
/content/bd_out.zip for `colab download`. Progress lines go to /content/bd.log.
"""
import os
import subprocess
import zipfile
from pathlib import Path

W = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
ARMS = []
for seed in (0, 1, 2):
    for init in ("e17c", "stock"):
        ARMS.append((f"spec-{init}-s{seed}", ["--init", init, "--seed", str(seed)]))
for init in ("e17c", "stock"):
    for w in W:
        tr = [x for x in W if x != w]
        for seed in (0, 1, 2):
            ARMS.append((f"lowo-{init}-{w}-s{seed}", ["--init", init, "--seed", str(seed), "--train", *tr, "--test", w]))
KEEP = [f"spec-{i}-s{s}" for i in ("e17c", "stock") for s in (0, 1, 2)]

BASE = Path("/content/exp")
D = BASE / "browser-decider"
LOG = open("/content/bd.log", "a")


def sh(cmd, **kw):
    LOG.write(f"$ {cmd}\n")
    LOG.flush()
    return subprocess.run(cmd, shell=True, stdout=LOG, stderr=subprocess.STDOUT, **kw)


if not D.exists():
    sh(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments {BASE}")
    sh("git sparse-checkout set browser-decider", cwd=BASE)
sh("git pull -q", cwd=BASE)
sh("pip install -q transformers==4.57.1")
ck = D / "out/base/st-ettin17m-tuned-e17-cap-c-epoch1"
if not ck.exists():
    ck.parent.mkdir(parents=True, exist_ok=True)
    sh(f"curl -sSL -o {ck}.zip https://github.com/oaustegard/experiments/releases/download/ettin-native-20261021/st-ettin17m-tuned-e17-cap-c-epoch1.zip && unzip -q -o {ck}.zip -d {ck}")
for name, args in ARMS:
    if (D / "results" / f"{name}.json").exists():
        continue
    r = sh(f"python3 run/train_enc.py {name} {' '.join(args)} 2>&1 | grep -v -i warning", cwd=D)
    LOG.write(f"ARM_DONE {name} rc={r.returncode}\n")
    LOG.flush()
with zipfile.ZipFile("/content/bd_out.zip", "w") as z:
    for f in (D / "results").glob("*.json"):
        z.write(f, f"results/{f.name}")
    for name in KEEP:
        for f in (D / "out/ckpt" / name).glob("*"):
            z.write(f, f"out/ckpt/{name}/{f.name}")
LOG.write("ALL_ARMS_DONE\n")
LOG.flush()
print("done")
