"""Embedder follow-up on the same VM, after run/colab_c.py: the arms that failed and the re-scores.

  68M joint and 68M contrastive-alone with gradient checkpointing (batch 24 ran out of memory on a 40 GB A100)
  the 68M joint model with its encoder rounded as packed
  the capemox baseline, loaded with Ettin's tokenizer
  every model scored before the final distractor pool was uploaded, scored again on it (suffix -p2, no BEIR)
Waits for colab_c.py and any embed.py process to finish first. Log: /content/dettin_c2.log, ends with ALL_DONE.
"""
import os
import subprocess
import time
import zipfile
from pathlib import Path

D = Path("/content/exp/dettin-68m")
LOG = open("/content/dettin_c2.log", "a")  # noqa: SIM115 (open for the whole run)
M68 = "jhu-clsp/ettin-encoder-68m"
E = "python3 -u run/embed.py"


def say(msg):
    LOG.write(msg + "\n")
    LOG.flush()


def busy():
    return any(subprocess.run(["pgrep", "-f", pat], capture_output=True, check=False).returncode == 0
               for pat in ("run/colab_c.py", "run/embed.py"))


def sh(cmd):
    say(f"$ {cmd}")
    r = subprocess.run(["bash", "-o", "pipefail", "-c", f"{cmd} 2>&1 | grep --line-buffered -v -i warning"],
                       stdout=LOG, stderr=subprocess.STDOUT, check=False, cwd=D)
    say(f"ARM_DONE rc={r.returncode} {cmd[:80]}")
    tmp = "/content/dettin_c.zip.part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (D / "results").glob("emb_*.json"):
            z.write(f, f"emb/{f.name}")
        for f in Path("/content/work/ckpt").glob("*.json"):
            z.write(f, f"emb_train/{f.name}")
    os.replace(tmp, "/content/dettin_c.zip")


def have(p):
    return Path(p).exists()


while busy():
    time.sleep(20)
say("STAGE gpu free")
if not have("/content/work/ckpt/e68-joint.pt"):
    sh(f"{E} train e68-joint --init {M68} --stage joint --from e68-distill --epochs 3 --bs 24 --lr 5e-5 --gc")
if not have(D / "results/emb_e68-joint.json"):
    sh(f"{E} eval e68-joint")
if not have("/content/work/ckpt/e68-contrast.pt"):
    sh(f"{E} train e68-contrast --init {M68} --stage contrast --epochs 3 --bs 24 --lr 5e-5 --gc")
if not have(D / "results/emb_e68-contrast.json"):
    sh(f"{E} eval e68-contrast")
if not have(D / "results/emb_e68-joint-q.json"):
    sh(f"{E} eval e68-joint --quant")
if not have(D / "results/emb_base-capemox.json"):
    sh(f"{E} eval --baseline capemox")
for name in ("e68-distill", "e17-distill", "e32-distill"):
    if not have(D / f"results/emb_{name}-p2.json"):
        sh(f"{E} eval {name} --no-beir --suffix=-p2")
for b in ("gte", "medcpt"):
    if not have(D / f"results/emb_base-{b}-p2.json"):
        sh(f"{E} eval --baseline {b} --no-beir --suffix=-p2")
if have("/content/work/ckpt/e68-joint.pt"):
    with zipfile.ZipFile("/content/dettin_c_e68.zip.part", "w", zipfile.ZIP_STORED) as z:
        z.write("/content/work/ckpt/e68-joint.pt", "e68-joint.pt")
        z.write("/content/work/ckpt/e68-joint.json", "e68-joint.json")
    os.replace("/content/dettin_c_e68.zip.part", "/content/dettin_c_e68.zip")
say("ALL_DONE")
