"""Run gliner-browser arms on a Colab GPU VM (started detached there by colab_drive.py):

    python3 colab_job.py --arms A0,A1,A2 [--baselines [B1_gliner_small-v2.1,...]]

Sparse-clones oaustegard/experiments (gliner-browser only) to /content/exp unless present, pulls,
installs the pinned library versions, runs run/prep_data.py (the benchmark zip is downloaded from
Google Drive on the VM and sha256-checked), then the requested baselines (bench.py) and arms
(train.py) one at a time from run/arms.json. Resume-safe: anything whose results/<name>.json exists
is skipped, and train.py itself resumes from out/<name>/ckpt. Every command runs under
`bash -o pipefail`, so a crash behind the warning filter is a non-zero rc. Progress goes to
/content/gliner.log. After every arm or baseline, results/*.json and each arm's progress.jsonl are
zipped to /content/gliner_out.zip; at the end the best checkpoints (out/<name>/best) go to
/content/gliner_ckpt.zip and the last log line is ALL_DONE.

--content, --no-setup and --arms-file exist so the runner can be exercised off-Colab.
"""
import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

PINS = "gliner==0.2.30 transformers==4.57.1 tokenizers==0.22.2 accelerate==1.15.0 huggingface_hub==0.36.2"
REPO = "https://github.com/oaustegard/experiments"

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="", help="comma list from arms.json, e.g. A0,A1")
ap.add_argument("--baselines", nargs="?", const="all", default="", help="no value = all baselines")
ap.add_argument("--content", type=Path, default=Path("/content"))
ap.add_argument("--no-setup", action="store_true", help="skip clone, pull and pip (use the checkout this file is in)")
ap.add_argument("--arms-file", type=Path, default=None)
ap.add_argument("--prep-args", default="")
a = ap.parse_args()

C = a.content
BASE = C / "exp"
D = Path(__file__).resolve().parents[1] if a.no_setup else BASE / "gliner-browser"
C.mkdir(parents=True, exist_ok=True)
LOG = open(C / "gliner.log", "a")


def mark(line):
    LOG.write(line + "\n")
    LOG.flush()


def sh(cmd, **kw):
    mark(f"$ {cmd}")
    # bash with pipefail, so a crash behind `| grep` is not reported as rc=0
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, **kw)


def flags(d):
    out = []
    for k, v in d.items():
        if k.startswith("--"):
            out += [k, str(v)]
    return out


def zip_results(arm_names):
    tmp = C / "gliner_out.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((D / "results").glob("*.json")):
            z.write(f, f"results/{f.name}")
        for n in arm_names:
            p = D / "out" / n / "progress.jsonl"
            if p.exists():
                z.write(p, f"out/{n}/progress.jsonl")
    os.replace(tmp, C / "gliner_out.zip")


def zip_ckpts(arm_names):
    tmp = C / "gliner_ckpt.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for n in arm_names:
            for f in sorted((D / "out" / n / "best").glob("*")):
                z.write(f, f"out/{n}/best/{f.name}")
    os.replace(tmp, C / "gliner_ckpt.zip")


def main():
    if not a.no_setup:
        if not D.exists():
            sh(f"git clone -q --depth 1 --filter=blob:none --sparse {REPO} {BASE}")
            sh("git sparse-checkout set gliner-browser", cwd=BASE)
        sh("git pull -q", cwd=BASE)
        sh("git log --oneline -1", cwd=BASE)
        r = sh(f"pip install -q {PINS}")
        if r.returncode:
            mark(f"PIP_FAILED rc={r.returncode}")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true")
    sh("python3 -c 'import torch, transformers, gliner; print(torch.__version__, transformers.__version__, gliner.__version__, torch.cuda.is_available())'")
    spec = json.loads((a.arms_file or D / "run" / "arms.json").read_text())
    arms = [x for x in a.arms.split(",") if x]
    bases = list(spec["baselines"]) if a.baselines == "all" else [x for x in a.baselines.split(",") if x]
    unknown = [x for x in arms if x not in spec["arms"]] + [x for x in bases if x not in spec["baselines"]]
    if unknown:
        mark(f"UNKNOWN {unknown}")
    (D / "results").mkdir(exist_ok=True)

    r = sh(f"python3 -u run/prep_data.py {a.prep_args} 2>&1 | grep --line-buffered -v -i warning", cwd=D)
    mark(f"PREP rc={r.returncode}")
    if r.returncode == 0:
        for name in bases:
            if name not in spec["baselines"] or (D / "results" / f"{name}.json").exists():
                continue
            model = spec["baselines"][name]["model"]
            r = sh(f"python3 -u run/bench.py --model {model} --out results/{name}.json "
                   f"2>&1 | grep --line-buffered -v -i warning", cwd=D)
            mark(f"BASELINE_DONE {name} rc={r.returncode}")
            zip_results(arms)
        for name in arms:
            if name not in spec["arms"] or (D / "results" / f"{name}.json").exists():
                continue
            args = flags({**spec["common"], **spec["arms"][name]})
            cmd = " ".join(["python3 -u run/train.py --name", name, *args])
            r = sh(f"{cmd} 2>&1 | grep --line-buffered -v -i warning", cwd=D)
            if r.returncode or not (D / "results" / f"{name}.json").exists():
                mark(f"ARM_FAILED {name} rc={r.returncode}")
            zip_results(arms)
    zip_ckpts(arms)
    mark("ALL_DONE")


try:
    main()
except Exception as e:      # the driver waits for ALL_DONE; say why it will not come
    mark(f"JOB_CRASHED {type(e).__name__}: {e}")
    raise
