"""Embedder scores on the paraphrased eval queries, on the same VM after run/colab_c2.py.

Every checkpoint and baseline is scored again with DETTIN_EMB=emb_data_para.json (same pool, queries
reworded by Sonnet to avoid the section's own terms), suffix -para, no BEIR.
Log: /content/dettin_c3.log, ends with ALL_DONE.
"""
import os
import subprocess
import time
import zipfile
from pathlib import Path

D = Path("/content/exp/dettin-68m")
LOG = open("/content/dettin_c3.log", "a")  # noqa: SIM115 (open for the whole run)
E = "DETTIN_EMB=emb_data_para.json python3 -u run/embed.py eval"


def say(msg):
    LOG.write(msg + "\n")
    LOG.flush()


def busy():
    return any(subprocess.run(["pgrep", "-f", pat], capture_output=True, check=False).returncode == 0
               for pat in ("run/colab_c.py", "run/colab_c2.py", "run/embed.py"))


while busy():
    time.sleep(20)
say("STAGE gpu free")
runs = [(f"{n}", f"{E} {n} --no-beir --suffix=-para") for n in
        ("e68-joint", "e32-joint", "e17-joint", "e68-distill", "e32-distill", "e17-distill", "e68-contrast")
        if Path(f"/content/work/ckpt/{n}.pt").exists()]
if Path("/content/work/ckpt/e68-joint.pt").exists():
    runs.append(("e68-joint-q", f"{E} e68-joint --quant --no-beir --suffix=-para"))
runs += [(f"base-{b}", f"{E} --baseline {b} --no-beir --suffix=-para") for b in ("gte", "bge-small", "medcpt", "capemox")]
for name, cmd in runs:
    if (D / "results" / f"emb_{name}-para.json").exists():
        continue
    say(f"$ {cmd}")
    r = subprocess.run(["bash", "-o", "pipefail", "-c", f"{cmd} 2>&1 | grep --line-buffered -v -i warning"],
                       stdout=LOG, stderr=subprocess.STDOUT, check=False, cwd=D)
    say(f"ARM_DONE rc={r.returncode} {name}")
    tmp = "/content/dettin_c.zip.part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (D / "results").glob("emb_*.json"):
            z.write(f, f"emb/{f.name}")
        for f in Path("/content/work/ckpt").glob("*.json"):
            z.write(f, f"emb_train/{f.name}")
    os.replace(tmp, "/content/dettin_c.zip")
say("ALL_DONE")
