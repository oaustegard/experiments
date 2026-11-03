"""Session side of the bridge runs: build a job's inputs, run it locally on SwiftShader (correctness
only), or submit it through the hub's scripts/gpu_bridge.py.

    python3 bridge.py check  --model export/base [--local]          # parity vs ort_gen.py
    python3 bridge.py bench  --model export/base --id ...           # decode ms/token, B=1 and B=8
    python3 bridge.py gen    --model export/base --name base --part 0/4 --id ...
    python3 bridge.py collect --name base                           # fetch gen results -> runs/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HUB = Path(os.environ.get("HUB", "/home/user/claude-workspace"))
GB = [sys.executable, str(HUB / "scripts/gpu_bridge.py")]
LOCAL = HERE / "bridge/local.js"
ORT_LOCAL = Path("/tmp/claude-0/ortweb/package/dist")
B_GEN = 8
MAX_NEW = {"mmlu": 1024, "gsm8k": 1024, "ifeval": 1536}


def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(HERE / "models/monad"))


def model_inputs(model: Path, d: Path, weights=True):
    """Files for graph, parts, meta, pieces; returns gpu_bridge --input specs and local inputs.json.
    weights=False sends only meta and pieces: the job then needs the session in ctx.cache from an
    earlier job in the same tab, and fails with a clear error otherwise."""
    meta = json.loads((model / "meta.json").read_text())
    spec, local = [], {}
    files = {"meta": model / "meta.json", "pieces": HERE / "export/pieces.json"}
    if weights:
        files["graph"] = model / "model.onnx"
        for i, p in enumerate(meta["parts"]):
            files[f"p{i}"] = model / p["file"]
    for k, f in files.items():
        spec.append(f"{k}={f}:u8")
        dst = d / f"{k}.bin"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(f, dst)
        local[k] = {"file": f"{k}.bin", "dtype": "u8"}
    return spec, local


def prompts_for(split="test"):
    from common import prompt
    from run_eval import question
    fx = json.loads((HERE / "data/fixture.json").read_text())[split]
    t = tok()
    return [(x["uid"], x["task"], t.encode(prompt(question(x)), add_special_tokens=False)) for x in fx]


def gen_batches(split="test", b=B_GEN):
    items = prompts_for(split)
    out = []
    for task in ("mmlu", "gsm8k", "ifeval"):
        xs = [x for x in items if x[1] == task]
        for i in range(0, len(xs), b):
            c = xs[i:i + b]
            out.append({"uids": [u for u, _, _ in c], "prompts": [p for _, _, p in c], "maxNew": MAX_NEW[task]})
    return out


def run(kind, a, cfg):
    d = HERE / "bridge/jobs" / a.id
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    spec, local = model_inputs(HERE / a.model, d, weights=not a.no_weights)
    if a.local:
        cfg["ort"] = "/ort/"
        local["cfg"] = {"json": cfg}
        (d / "inputs.json").write_text(json.dumps(local))
        if not (d / "ort").exists():
            os.symlink(ORT_LOCAL, d / "ort")
        r = subprocess.run(["node", str(LOCAL), str(HERE / "bridge/job.js"), str(d), str(d / "out.json")],
                           cwd=str(LOCAL.parent), capture_output=True, text=True, timeout=a.timeout // 1000 + 600)
        print(r.stdout[-3000:], r.stderr[-3000:])
        return
    (d / "cfg.json").write_text(json.dumps(cfg))
    cmd = GB + ["submit", str(HERE / "bridge/job.js"), "--id", a.id, "--note", a.note or kind,
                "--timeout", str(a.timeout), "--input", f"cfg=@{json.dumps(cfg)}"]
    for s in spec:
        cmd += ["--input", s]
    if a.target:
        cmd += ["--target", a.target]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["check", "bench", "gen", "collect"])
    ap.add_argument("--model", default="export/base")
    ap.add_argument("--id")
    ap.add_argument("--note")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--timeout", type=int, default=600000)
    ap.add_argument("--target", default="Mac")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--part", default="0/1", help="k/n: this job takes batches k, k+n, ...")
    ap.add_argument("--batches", default="", help="explicit comma list of batch indexes")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ep", default="")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--opt", default="")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if a.kind == "check":
        t = tok()  # noqa
        from common import prompt
        qs = json.loads((HERE / "bridge/check_prompts.json").read_text())
        cfg = {"mode": "check", "steps": a.steps, "prompts": [t.encode(prompt(q), add_special_tokens=False) for q in qs]}
    elif a.kind == "bench":
        t = tok()
        from common import prompt
        cfg = {"mode": "bench", "steps": a.steps if a.local else 128, "runs": 1 if a.local else 3, "batches": [1] if a.local else [1, 8],
               "prompt": t.encode(prompt("Explain why the sky is blue in two sentences."), add_special_tokens=False)}
    elif a.kind == "gen":
        bs = gen_batches(a.split)
        if a.batches:
            idx = [int(x) for x in a.batches.split(",")]
        else:
            k, n = map(int, a.part.split("/"))
            idx = list(range(k, len(bs), n))
        cfg = {"mode": "gen", "budgetMs": a.timeout - 90000, "batches": [bs[i] for i in idx], "batchIndex": idx}
    else:
        raise SystemExit("collect: see collect.py")
    if a.ep:
        cfg["ep"] = a.ep
    if a.opt:
        cfg["opt"] = a.opt
    if a.cache_tag:
        cfg["cacheTag"] = a.cache_tag
    run(a.kind, a, cfg)


if __name__ == "__main__":
    main()
