"""Build kaggle/job_<name>.py: runs the SciFact check (scifact.py) for all four variants on Kaggle's 2x T4.

    python3 kaggle_job.py --limit 64 --name smoke     # cut-down run first (docs/kaggle.md)
    python3 kaggle_job.py --name full
    python3 <claude-workspace>/scripts/kaggle_job.py push sparseup-scifact-full kaggle/job_full.py --accel gpu --internet

The job embeds ref.py, pack.py and scifact.py, downloads the model and SciFact from the Hub, builds the
int8 and q4 packs with pack.py, encodes fp32 + int8 on GPU 0 and q4 + bf16 on GPU 1, scores, and leaves
results/scifact.json and the encodings under /kaggle/working/.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNNER = r'''
import os, subprocess, sys, threading
from pathlib import Path
W = Path("/kaggle/working"); S = W / "src"; S.mkdir(parents=True, exist_ok=True)
for name, src in SOURCES.items():
    (S / name).write_text(src)
from huggingface_hub import hf_hub_download
for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
    hf_hub_download("Linkup-Platform/linkup-sparseup-embed-v2", f, local_dir=S / "hf")
for repo, f in [("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet"),
                ("BeIR/scifact", "queries/queries-00000-of-00001.parquet"), ("BeIR/scifact-qrels", "test.tsv")]:
    hf_hub_download(repo, f, repo_type="dataset", local_dir=S / "cache" / "scifact")
print("downloads done", flush=True)
for q in ("int8", "q4"):
    subprocess.run([sys.executable, "pack.py", "--hf", "hf", "--out", f"packs/{q}", "--quant", q], cwd=S, check=True)
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)

def lane(gpu, variants):
    for v in variants:
        cmd = [sys.executable, "-u", "scifact.py", "encode", "--variant", v, "--device", f"cuda:{gpu}"]
        if LIMIT:
            cmd += ["--limit", str(LIMIT)]
        p = subprocess.Popen(cmd, cwd=S, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if "warn" not in line.lower():
                print(f"[gpu{gpu}] {line}", end="", flush=True)
        if p.wait():
            print(f"[gpu{gpu}] {v} FAILED rc={p.returncode}", flush=True)

ts = [threading.Thread(target=lane, args=(0, ["fp32", "int8"])), threading.Thread(target=lane, args=(1, ["q4", "bf16"]))]
[t.start() for t in ts]; [t.join() for t in ts]
if LIMIT:
    for v in ("fp32", "int8", "q4", "bf16"):
        a, b = S / "cache" / f"scifact_{v}_limit{LIMIT}.npz", S / "cache" / f"scifact_{v}.npz"
        if a.exists():
            a.rename(b)
    src = (S / "scifact.py").read_text().replace("n_expected_docs = len(corpus_doc_ids)", f"n_expected_docs = {LIMIT}")
    (S / "scifact.py").write_text(src)
subprocess.run([sys.executable, "scifact.py", "score"], cwd=S, check=True)
import shutil
(W / "out").mkdir(exist_ok=True)
shutil.copy(S / "results" / "scifact.json", W / "out" / "scifact.json")
for v in ("fp32", "int8", "q4", "bf16"):
    f = S / "cache" / f"scifact_{v}.npz"
    if f.exists():
        shutil.copy(f, W / "out" / f.name)
shutil.rmtree(S / "hf"); shutil.rmtree(S / "packs")
print("done", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="full")
    a = ap.parse_args()
    sources = {n: (HERE / n).read_text() for n in ("ref.py", "pack.py", "scifact.py")}
    out = HERE / "kaggle" / f"job_{a.name}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"SOURCES = {sources!r}\nLIMIT = {a.limit}\n" + RUNNER)
    print(out)


if __name__ == "__main__":
    main()
