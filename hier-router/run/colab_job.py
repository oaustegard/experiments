"""Run the GPU part inside a Colab kernel.

    colab new -s hr --gpu L4
    colab exec -s hr --timeout 7200 -f run/colab_job.py
    colab download -s hr /content/hr_out.zip hier-router/out/hr_out.zip

Clones the repo at main (sparse), fetches MASSIVE en-US parquet from the Hub's parquet export, retrains
adapter-library's 12 family sub-spikes (round 4's R-sub arms; their probabilities were never committed),
then runs train_eval.py and analyze.py. Progress goes to /content/hr.log.
"""
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

BASE = Path("/content/exp")
HR, AL = BASE / "hier-router", BASE / "adapter-library"
LOG = open("/content/hr.log", "a")


def sh(cmd, **kw):
    LOG.write(f"$ {cmd}\n")
    LOG.flush()
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=LOG, stderr=subprocess.STDOUT, **kw)


if not HR.exists():
    sh(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments {BASE}")
    sh("git sparse-checkout set hier-router adapter-library browser-decider/run browser-decider/data _lib", cwd=BASE)
sh("git pull -q", cwd=BASE)
sh("pip install -q transformers==4.57.1")
md = HR / "data" / "massive"
md.mkdir(parents=True, exist_ok=True)
api = "https://huggingface.co/api/datasets/AmazonScience/massive/tree/refs%2Fconvert%2Fparquet/en-US/"
for split in ("train", "test"):
    if not (md / f"{split}.parquet").exists():
        f = json.loads(urllib.request.urlopen(api + split).read())[0]["path"]
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/AmazonScience/massive/resolve/refs%2Fconvert%2Fparquet/{f}",
                                   md / f"{split}.parquet")
ARGS = Path("/content/hr_args.txt").read_text() if Path("/content/hr_args.txt").exists() else ""
if "--round" not in ARGS and len(list((AL / "out" / "probs").glob("R-sub-*.json"))) < 12:  # rounds 2+ reuse round 1's
    sh("rm -f results/R-sub-*.json", cwd=AL)
    r = sh("python3 -I run/lora.py --only R-sub 2>&1 | grep --line-buffered -v -i warning", cwd=AL)
    LOG.write(f"SUB_RC {r.returncode}\n")
import os
extra = os.environ.get("HR_ARGS", Path("/content/hr_args.txt").read_text().strip() if Path("/content/hr_args.txt").exists() else "")
r = sh(f"python3 -I run/train_eval.py {extra} 2>&1 | grep --line-buffered -v -i warning", cwd=HR)
LOG.write(f"TRAIN_EVAL_RC {r.returncode}\n")
r = sh("python3 -I run/analyze.py", cwd=HR)
LOG.write(f"ANALYZE_RC {r.returncode}\n")
with zipfile.ZipFile("/content/hr_out.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in list((HR / "results").glob("*.json")) + list((HR / "out").glob("*.json")) + list((HR / "out" / "adapters").glob("*.pt")):
        z.write(f, f"hier-router/{f.relative_to(HR)}")
    for f in (AL / "out" / "probs").glob("R-sub-*.json"):
        z.write(f, f"adapter-library/{f.relative_to(AL)}")
LOG.write("ALL_DONE\n")
LOG.flush()
print("done")
