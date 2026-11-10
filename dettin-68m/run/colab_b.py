"""Part C decisions on a Colab GPU VM: the life-science rater and classifier at 17M, 32M and 68M.

Expects /content/data/cases_haiku.jsonl and cases_jev.jsonl (uploaded by run/colab_put.py).
  full fine-tune on Haiku's labels, sizes 17/32/68, seeds 0-2 (learning rates from Part A)
  68M on Jev's labels, seed 0 (the teacher comparison)
  three LoRA spikes on the frozen stock 68M base, seeds 0 and 1
  packed variants of each size's seed-0 model, scored on the same test cases
Results (summaries and per-question predictions) are re-zipped to /content/dettin_b.zip after each arm.
Log: /content/dettin_b.log, ends with ALL_DONE.
"""
import os
import subprocess
import zipfile
from pathlib import Path

BASE = Path("/content/exp")
BD, D = BASE / "browser-decider", BASE / "dettin-68m"
DATA, SPK = Path("/content/data"), Path("/content/spikes_ls")
LOG = open("/content/dettin_b.log", "a")  # noqa: SIM115 (open for the whole run)
INIT = {"17": ("stock", "8e-5"), "32": ("stock32", "5e-5"), "68": ("stock68", "8e-5")}
EPOCHS = os.environ.get("DETTIN_EPOCHS", "2")


def say(msg):
    LOG.write(msg + "\n")
    LOG.flush()


def sh(cmd, **kw):
    say(f"$ {cmd}")
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, check=False, **kw)


def rezip():
    tmp = "/content/dettin_b.zip.part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (BD / "results").glob("*b-*.json"):
            z.write(f, f"ls/{f.name}")
        for f in SPK.glob("*.json"):
            z.write(f, f"ls_spikes/{f.name}")
    os.replace(tmp, "/content/dettin_b.zip")


def train(name, size, seed, teacher):
    if not (BD / "results" / f"{name}.json").exists():
        init, lr = INIT[size]
        r = sh(f"python3 -u run/train_enc.py {name} --init {init} --lr {lr} --seed {seed} --data {DATA}/cases_{teacher}.jsonl "
               f"--amp --epochs {EPOCHS} --max-len 1024 2>&1 | grep --line-buffered -v -i warning", cwd=BD)
        say(f"ARM_DONE {name} rc={r.returncode}")
        rezip()


sh("pip install -q transformers==4.57.1 brotli")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
for size in ("68", "17", "32"):
    train(f"b-{size}-haiku-s0", size, 0, "haiku")
train("b-68-jev-s0", "68", 0, "jev")
r = sh(f"AL_BASE=jhu-clsp/ettin-encoder-68m AL_AMP=1 python3 -u run/spike_ls.py {DATA}/cases_haiku.jsonl {SPK} --seeds 0 1 "
       f"--epochs {EPOCHS} 2>&1 | grep --line-buffered -v -i warning", cwd=D)
say(f"SPIKES_DONE rc={r.returncode}")
rezip()
names = [f"b-{s}-haiku-s0" for s in ("17", "32", "68")]
todo = [n for n in names if not (BD / "results" / f"quant_levels-{n}.json").exists()]
if todo:
    r = sh(f"python3 -u run/quant_levels.py --data {DATA}/cases_haiku.jsonl {' '.join(todo)} 2>&1 | grep --line-buffered -v -i warning", cwd=BD)
    say(f"QUANT_DONE rc={r.returncode}")
    rezip()
for size in ("68", "17", "32"):
    for seed in (1, 2):
        train(f"b-{size}-haiku-s{seed}", size, seed, "haiku")
with zipfile.ZipFile("/content/dettin_b_adapters.zip.part", "w", zipfile.ZIP_STORED) as z:
    for f in (SPK / "adapters").glob("*-s0.pt"):
        z.write(f, f"adapters_ls/{f.name}")
os.replace("/content/dettin_b_adapters.zip.part", "/content/dettin_b_adapters.zip")
say("ALL_DONE")
