"""Forward-pass speed of the stock Ettin encoders at 17M, 32M and 68M.

    python3 run/speed.py --tag cpu4            # this machine; fp32 on CPU, bf16 autocast on CUDA
Random token ids at 128, 512 and 1,024 tokens, batch 1 and batch 32 (GPU) or batch 1 and 8 (CPU).
Median of N timed passes after two warm-up passes. Writes results/speed_<tag>.json.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"17m": "jhu-clsp/ettin-encoder-17m", "32m": "jhu-clsp/ettin-encoder-32m", "68m": "jhu-clsp/ettin-encoder-68m"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    torch.set_num_threads(a.threads)
    n = a.n or (20 if cuda else 7)
    res = {"device": torch.cuda.get_device_name(0) if cuda else f"cpu x{a.threads}", "rows": []}
    for name, mid in MODELS.items():
        model = AutoModel.from_pretrained(mid, attn_implementation="sdpa").to(dev).eval()
        for L in (128, 512, 1024):
            for bs in ((1, 32) if cuda else (1, 8)):
                ids = torch.randint(1000, 30000, (bs, L), device=dev)
                att = torch.ones_like(ids)
                ts = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    for i in range(n + 2):
                        t0 = time.perf_counter()
                        model(input_ids=ids, attention_mask=att).last_hidden_state.float().mean().item()
                        if cuda:
                            torch.cuda.synchronize()
                        if i >= 2:
                            ts.append(time.perf_counter() - t0)
                ms = statistics.median(ts) * 1000
                row = {"model": name, "tokens": L, "batch": bs, "ms_per_batch": round(ms, 2),
                       "seqs_per_s": round(bs / ms * 1000, 1)}
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
        del model
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"speed_{a.tag}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
