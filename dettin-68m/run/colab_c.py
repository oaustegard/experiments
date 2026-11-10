"""Part C embedder on a Colab GPU VM. Expects /content/data/sections.jsonl.gz and emb_data.json.

  teacher vectors (gte-modernbert-base) for every section and training query
  per size 17/32/68: distill (teacher pull alone)
  baselines on the same eval path: the teacher, capemox 68M, MedCPT, bge-small
  then, once the Jev-filtered negatives are uploaded: joint (contrastive + pull) from each distill
  checkpoint; at 68M also contrastive alone from the stock encoder, and the joint model with its
  encoder rounded as packed
Each eval writes results/emb_<name>.json; they are re-zipped to /content/dettin_c.zip as they land.
Log: /content/dettin_c.log, ends with ALL_DONE.
"""
import os
import subprocess
import time
import zipfile
from pathlib import Path

D = Path("/content/exp/dettin-68m")
LOG = open("/content/dettin_c.log", "a")  # noqa: SIM115 (open for the whole run)
MODELS = {"68": "jhu-clsp/ettin-encoder-68m", "17": "jhu-clsp/ettin-encoder-17m", "32": "jhu-clsp/ettin-encoder-32m"}


def say(msg):
    LOG.write(msg + "\n")
    LOG.flush()


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
    return r.returncode


def have(p):
    return Path(p).exists()


E = "python3 -u run/embed.py"
sh("pip install -q transformers==4.57.1 datasets")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
if not have("/content/work/teacher_queries.npy"):
    sh(f"{E} teacher")
if not have(D / "results/emb_base-gte.json"):
    sh(f"{E} eval --baseline gte")
for size in ("68", "17", "32"):
    if not have(f"/content/work/ckpt/e{size}-distill.pt"):
        sh(f"{E} train e{size}-distill --init {MODELS[size]} --stage distill --epochs 2 --bs 64 --lr 1e-4")
    if not have(D / f"results/emb_e{size}-distill.json"):
        sh(f"{E} eval e{size}-distill" + ("" if size == "68" else " --no-beir"))
for b in ("capemox", "medcpt", "bge-small"):
    if not have(D / f"results/emb_base-{b}.json"):
        sh(f"{E} eval --baseline {b}")
# The contrastive stages need emb_data.json with Jev-filtered negatives; the local side uploads it and
# then /content/data/emb_ready. The stages above do not read the negatives, so they run first.
while not have("/content/data/emb_ready"):
    time.sleep(20)
say("STAGE negatives ready")
for size in ("68", "17", "32"):
    if not have(f"/content/work/ckpt/e{size}-joint.pt"):
        sh(f"{E} train e{size}-joint --init {MODELS[size]} --stage joint --from e{size}-distill --epochs 3 --bs 24 --lr 5e-5")
    if not have(D / f"results/emb_e{size}-joint.json"):
        sh(f"{E} eval e{size}-joint")
if not have("/content/work/ckpt/e68-contrast.pt"):
    sh(f"{E} train e68-contrast --init {MODELS['68']} --stage contrast --epochs 3 --bs 24 --lr 5e-5")
if not have(D / "results/emb_e68-contrast.json"):
    sh(f"{E} eval e68-contrast")
if not have(D / "results/emb_e68-joint-q.json"):
    sh(f"{E} eval e68-joint --quant")
with zipfile.ZipFile("/content/dettin_c_e68.zip.part", "w", zipfile.ZIP_STORED) as z:
    z.write("/content/work/ckpt/e68-joint.pt", "e68-joint.pt")
    z.write("/content/work/ckpt/e68-joint.json", "e68-joint.json")
os.replace("/content/dettin_c_e68.zip.part", "/content/dettin_c_e68.zip")
say("ALL_DONE")
