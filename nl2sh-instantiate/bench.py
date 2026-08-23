#!/usr/bin/env python3
"""The laptop bar from issue #52, measured rather than quoted.

Four numbers, on whatever box this runs on: time to first token, decode rate at
batch=1, peak resident memory, and total wall-clock for one request. The issue
asks for the decode roofline check first — `bandwidth / weight_bytes` is the
hard single-stream ceiling — so the measured rate is printed beside the bytes
that have to move per token, and a rate far under the roofline means the
bottleneck is elsewhere (here: fp32 weights on 4 CPU cores, no quantisation).

Retrieval is not included. Stage 1's retrieval tier answers in ~200 ms and is a
separate artifact; this measures the generator alone.

    python3 bench.py --model unsloth/gemma-3-270m-it --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "nl2sh-retrieval"))

import prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="weight dtype. stage 1 ran float32; a 4B model does not fit 15 GB at float32.")
    ap.add_argument("--condition", default="instantiate", choices=sorted(prompts.BUILDERS))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=HERE / "results_bench.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Captured before the model loads: a run started on a busy box is not a
    # batch=1 measurement, and the reader deserves to see that in the row.
    load1 = os.getloadavg()[0]
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
    load_s = time.perf_counter() - t_load
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    srcs = ["find — Find files by extension: find root_path -name '*.ext'",
            "grep — Search for a pattern: grep 'search_pattern' path/to/file",
            "tar — Create an archive: tar cf path/to/target.tar file1 file2"]
    reqs = [f"Find every .log file under /var/log{'' if i == 0 else ' from the last %d days' % i}"
            for i in range(a.n)]

    ttfts, rates, walls = [], [], []
    for i, nl in enumerate(reqs):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompts.BUILDERS[a.condition](nl, srcs)}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            one = model.generate(**ids, max_new_tokens=1, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ttft = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        wall = time.perf_counter() - t1
        new = int(out.shape[1] - ids["input_ids"].shape[1])
        if i == 0:
            continue  # first pass warms caches; not a steady-state measurement
        ttfts.append(ttft)
        walls.append(wall)
        rates.append(new / wall)
        del one

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    mean = lambda xs: sum(xs) / len(xs)
    summary = {
        # Thread count and load average belong in the artifact, not only in
        # whatever prose quotes it. Four bench rows shipped with "batch=1 on 4
        # CPU cores" in their commit message while actually running pinned to
        # two threads with another model decoding beside them; nothing in the
        # JSON contradicted the claim, so nothing caught it.
        "threads": int(os.environ.get("OMP_NUM_THREADS", 0)) or os.cpu_count(),
        "cpu_count": os.cpu_count(),
        "load1_at_start": round(load1, 2),
        "model": a.model, "condition": a.condition, "dtype": a.dtype,
        "params": n_params, "weight_bytes": weight_bytes,
        "weight_gib": round(weight_bytes / 2**30, 3),
        "load_seconds": round(load_s, 1),
        "peak_rss_mib": round(rss_mb, 0),
        "n_measured": len(rates),
        "ttft_ms_mean": round(mean(ttfts) * 1000, 1),
        "ttft_ms_min": round(min(ttfts) * 1000, 1),
        "decode_tok_per_s_mean": round(mean(rates), 1),
        "decode_tok_per_s_max": round(max(rates), 1),
        "wall_seconds_64_tokens_mean": round(mean(walls), 2),
        "roofline_note": ("single-stream ceiling is memory_bandwidth / weight_bytes; "
                          f"at 100 GB/s this {a.dtype} copy tops out near "
                          f"{round(100e9 / weight_bytes)} tok/s, and a 4-bit copy near "
                          f"{round(100e9 / (n_params * 0.5))} tok/s"),
    }
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
