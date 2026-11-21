"""Kaggle job: stage 2, the training-free loop screen, on T4 x2.

Fetches monad-looped and monad-weight-edit's fixture from oaustegard/experiments at a pinned commit,
downloads PleIAs/Monad, then on each GPU runs tf_eval.py (teacher-forced metrics, SYNTH test rows) and
gen_eval.py (greedy generation, fixture dev rows) over its share of configs_screen.json. Outputs land in
/kaggle/working/{results,runs}. Worker output is passed through to stdout, prefixed by worker id.
COMMIT and SHORT are substituted by launch.py before pushing.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

COMMIT = "__COMMIT__"
SHORT = __SHORT__
W = Path("/kaggle/working")
SRC = Path("/kaggle/src")


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def fetch():
    sh(f"git clone --filter=blob:none --no-checkout https://github.com/oaustegard/experiments {SRC}")
    sh(f"cd {SRC} && git sparse-checkout init --cone && git sparse-checkout set monad-looped monad-weight-edit/data "
       f"&& git checkout {COMMIT}")
    sh(f"cd {SRC} && git show {COMMIT}:monad-weight-edit/summarize.py > monad-weight-edit/summarize.py && "
       f"git show {COMMIT}:monad-weight-edit/common.py > monad-weight-edit/common.py")
    sh("pip install -q safetensors transformers 2>&1 | tail -1")
    sh(f"python -c \"from huggingface_hub import snapshot_download as s; s('PleIAs/Monad', local_dir='{SRC}/monad-looped/models/monad', "
       f"allow_patterns=['*.json','*.safetensors'])\"")


def run_worker(w, n):
    d = SRC / "monad-looped"
    cfgs = json.loads((d / "configs_screen.json").read_text())
    base, rest = cfgs[0], cfgs[1:]
    mine = rest[w::n]
    (d / f"cfg_w{w}.json").write_text(json.dumps([base] + mine))
    gen = ([base] if w == 0 else []) + mine
    (d / f"gen_w{w}.json").write_text(json.dumps(gen))
    lim_tf = "--limit 8" if SHORT else ""
    lim_gen = "--limit 4 --max-new-scale 0.25" if SHORT else ""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w), PYTHONUNBUFFERED="1")
    cmd = (f"cd {d} && python tf_eval.py --configs cfg_w{w}.json --split test {lim_tf} --out {W}/results/tf_screen_w{w}.json && "
           f"python gen_eval.py --configs gen_w{w}.json --split dev {lim_gen} --runs {W}/runs")
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "Warning" not in line:
            print(f"[w{w}] {line}", end="", flush=True)
    return p.wait()


def main():
    (W / "results").mkdir(parents=True, exist_ok=True)
    fetch()
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    import torch
    n = max(1, torch.cuda.device_count())
    rcs = [None] * n
    ts = [threading.Thread(target=lambda i=i: rcs.__setitem__(i, run_worker(i, n))) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print("worker exit codes", rcs, flush=True)
    if any(rcs):
        sys.exit(1)
    print("done screen", flush=True)


if __name__ == "__main__":
    main()
