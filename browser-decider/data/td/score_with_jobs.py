"""Score a decision model on the Typed Decisions test split with HF Jobs.

Serves the model with vLLM's `/v1/systemone` (vLLM nightly), sends one request
per case (state + all five questions), and reports accuracy, KL from gold,
Brier and ECE over the full distributions.

Usage (inside the job, after setup): python bench/score_with_jobs.py MODEL_ID
Optional env: SERVE_FLAGS (extra `vllm serve` flags).
Results are printed and saved to results.json.
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time

import requests
from datasets import load_dataset

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"


def score_one(pred, gold):
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    if gold["type"] == "noul":  # engines report yes/no; gold uses true/false
        pred = dict(pred)
        pred["probabilities"] = {
            {"yes": "true", "no": "false"}.get(k, k): v for k, v in pred["probabilities"].items()
        }
    p = [float(pred["probabilities"].get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    am_p = p.index(max(p))
    am_g = g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return am_p == am_g, kl, brier, max(p)


def main():
    model = sys.argv[1]
    flags = os.environ.get("SERVE_FLAGS", "--max-model-len 8192 --max-num-seqs 8 --logprobs-mode raw_logprobs")
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--port", str(PORT)] + flags.split(),
        stdout=sys.stdout, stderr=sys.stdout,
    )
    while True:
        if proc.poll() is not None:
            raise SystemExit(f"vllm serve exited with code {proc.returncode}")
        try:
            if requests.get(BASE + "/health", timeout=5).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    print(f"[bench] server ready for {model}", flush=True)

    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    accs, kls, briers, confs = [], [], [], []
    for i, raw in enumerate(ds):
        payload = {
            "model": model,
            "state": json.loads(raw["state"]),
            "questions": json.loads(raw["questions"]),
        }
        r = requests.post(BASE + "/v1/systemone", json=payload, timeout=300)
        r.raise_for_status()
        answers = r.json()["answers"]
        gold = json.loads(raw["gold"])
        for qid, gq in gold.items():
            ok, kl, brier, conf = score_one(answers.get(qid, {"probabilities": {}}), gq)
            accs.append(ok)
            kls.append(kl)
            briers.append(brier)
            confs.append(conf)
        if (i + 1) % 50 == 0:
            print(f"[bench] {i + 1}/{len(ds)} cases (acc {sum(accs) / len(accs):.4f})", flush=True)

    n = len(accs)
    n_bins = 10
    bin_hits = [0] * n_bins
    bin_total = [0] * n_bins
    bin_conf = [0.0] * n_bins
    for ok, conf in zip(accs, confs):
        b = min(int(conf * n_bins), n_bins - 1)
        bin_total[b] += 1
        bin_hits[b] += int(ok)
        bin_conf[b] += conf
    ece = 0.0
    for b in range(n_bins):
        if bin_total[b]:
            ece += bin_total[b] / n * abs(bin_hits[b] / bin_total[b] - bin_conf[b] / bin_total[b])
    summary = {
        "model": model,
        "n_cases": len(ds),
        "n_decisions": n,
        "accuracy": sum(accs) / n,
        "kl_from_gold": sum(kls) / n,
        "brier": sum(briers) / n,
        "ece": ece,
    }
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("[bench] open a discussion on the dataset with these numbers and the mode (specialist or general).", flush=True)
    proc.terminate()


if __name__ == "__main__":
    main()