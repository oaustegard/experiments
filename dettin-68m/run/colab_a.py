"""Part A on a Colab GPU VM: the dEttin recipe (Typed Decisions specialist, packed builds, LoRA spikes)
at 17M, 32M and 68M under one code path. Started detached by run/colab_drive.py.

Stages, each resume-safe (an arm whose results file exists is skipped):
  A0  learning rate for 32M and 68M by validation KL, seed 0, from LRS
  A1  seeds 1 and 2 at the chosen rate; 17M seeds 0-2 at 8e-5
  A2  leave-one-workflow-out at 68M
  A3  packed variants (fp32, t4-w8L63, t4-w8L31) of each size's seed-0 checkpoint
  A4  LoRA r8 spike vs full fine-tune per workflow at 68M (adapter-library S arms), 3 seeds
  A5  GPU speed
Results are re-zipped to /content/dettin_a.zip after every arm; the packed 68M build goes to
/content/dettin_a_packed68.zip. Progress lines go to /content/dettin_a.log, which ends with ALL_DONE.
"""
import json
import os
import subprocess
import zipfile
from pathlib import Path

BASE = Path("/content/exp")
BD, AL, D = BASE / "browser-decider", BASE / "adapter-library", BASE / "dettin-68m"
ALOUT = Path("/content/al68")
LOG = open("/content/dettin_a.log", "a")  # noqa: SIM115 (open for the whole run)
W = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
INIT = {"17": "stock", "32": "stock32", "68": "stock68"}
LRS = ["3e-5", "5e-5", "8e-5"]


def say(msg):
    LOG.write(msg + "\n")
    LOG.flush()


def sh(cmd, **kw):
    say(f"$ {cmd}")
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, check=False, **kw)


def rezip():
    tmp = "/content/dettin_a.zip.part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (BD / "results").glob("*a-*.json"):
            z.write(f, f"td/{f.name}")
        for f in (ALOUT / "results").glob("*.json"):
            z.write(f, f"spikes/{f.name}")
        for f in (D / "results").glob("speed_*.json"):
            z.write(f, f"speed/{f.name}")
        if Path("/content/a_lr.json").exists():
            z.write("/content/a_lr.json", "td/a_lr.json")
    os.replace(tmp, "/content/dettin_a.zip")


def train(name, args):
    if not (BD / "results" / f"{name}.json").exists():
        r = sh(f"python3 -u run/train_enc.py {name} {args} --amp --epochs 10 2>&1 | grep --line-buffered -v -i warning", cwd=BD)
        say(f"ARM_DONE {name} rc={r.returncode}")
        rezip()
    f = BD / "results" / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


sh("pip install -q transformers==4.57.1 brotli")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

# A0: learning rate
chosen = {"17": "8e-5"}
for size in ("32", "68"):
    val = {}
    for lr in LRS:
        res = train(f"a-{size}-lr{lr}-s0", f"--init {INIT[size]} --lr {lr} --seed 0")
        if res:
            val[lr] = min(e["val_kl"] for e in res["log"])
    chosen[size] = min(val, key=val.get) if val else "8e-5"
    say(f"LR_CHOSEN {size} {chosen[size]} {json.dumps(val)}")
Path("/content/a_lr.json").write_text(json.dumps(chosen))

# A1: seeds
for size in ("17", "32", "68"):
    for seed in (0, 1, 2):
        train(f"a-{size}-lr{chosen[size]}-s{seed}", f"--init {INIT[size]} --lr {chosen[size]} --seed {seed}")

# A2: leave one workflow out, 68M
for w in W:
    others = " ".join(x for x in W if x != w)
    train(f"a-68-lowo-{w}", f"--init stock68 --lr {chosen['68']} --seed 0 --train {others} --test {w}")

# A3: packed variants of each size's seed-0 checkpoint
names = [f"a-{s}-lr{chosen[s]}-s0" for s in ("17", "32", "68")]
todo = [n for n in names if not (BD / "results" / f"quant_levels-{n}.json").exists()]
if todo:
    r = sh(f"python3 -u run/quant_levels.py {' '.join(todo)} 2>&1 | grep --line-buffered -v -i warning", cwd=BD)
    say(f"QUANT_DONE rc={r.returncode}")
    rezip()
n68 = names[2]
if not Path("/content/dettin_a_packed68.zip").exists():
    r = sh(f"python3 -u run/pack.py {n68} --variant t4-w8L63 --check 4 2>&1 | grep --line-buffered -v -i warning", cwd=BD)
    out = BD / "out" / "web" / f"{n68}-t4-w8L63"
    if r.returncode == 0 and out.exists():
        with zipfile.ZipFile("/content/dettin_a_packed68.zip.part", "w", zipfile.ZIP_STORED) as z:
            for f in out.glob("*"):
                z.write(f, f"{out.name}/{f.name}")
        os.replace("/content/dettin_a_packed68.zip.part", "/content/dettin_a_packed68.zip")
    say(f"PACK_DONE rc={r.returncode}")

# A4: LoRA spikes vs full fine-tunes on the 68M base
env = f"AL_BASE=jhu-clsp/ettin-encoder-68m AL_OUT_ROOT={ALOUT} AL_LR_FULL={chosen['68']} AL_AMP=1"
r = sh(f"{env} python3 -u run/lora.py --only S- 2>&1 | grep --line-buffered -v -i warning", cwd=AL)
say(f"SPIKES_DONE rc={r.returncode}")
rezip()
with zipfile.ZipFile("/content/dettin_a_adapters.zip.part", "w", zipfile.ZIP_STORED) as z:
    for f in (ALOUT / "out" / "adapters").glob("S-lora-*-s0.pt"):
        z.write(f, f"adapters68/{f.name}")
os.replace("/content/dettin_a_adapters.zip.part", "/content/dettin_a_adapters.zip")

# A5: GPU speed
r = sh("python3 -u run/speed.py --tag gpu 2>&1 | grep --line-buffered -v -i warning", cwd=D)
say(f"SPEED_DONE rc={r.returncode}")
rezip()
say("ALL_DONE")
