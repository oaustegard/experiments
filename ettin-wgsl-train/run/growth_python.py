"""adapter-library's warm-start growth arms with more seeds, for the WGSL-vs-PyTorch comparison.

    python3 -I run/growth_python.py OUT_DIR [SEEDS e.g. 0-9] [SIZES e.g. 64,300]

The PyTorch side of decider-lora-creator's growth comparison: invoice_processing held out, the start
is the merge of the other three workflows' round-1 LoRA spikes (S-lora-<w>-s0, rebuilt first if
adapter-library/out/adapters lacks them), then `revealed(w, seed, n)` cases for `growth_epochs(n)`
epochs. This is lora.py's "grow" branch line for line, for seeds lora.py does not list. Each arm
writes OUT_DIR/G-warm-<w>-n<n>-s<seed>.json in lora.py's format; existing files are skipped.
Runs on a GPU (run/colab_growth.py); about 10 s (n = 64) and 40 s (n = 300) an arm on an L4.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

AL = experiment("adapter-library")
sys.path[:0] = [str(AL / "run"), str(experiment("browser-decider") / "run")]
import lora as L  # noqa: E402
from shared import WORKFLOWS, load_cases, revealed  # noqa: E402

W = "invoice_processing"
SMOKE = int(__import__("os").environ.get("GROWTH_SMOKE", "0"))   # max steps per training run, for a CPU check


def spans(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    out = Path(sys.argv[1])
    seeds = spans(sys.argv[2]) if len(sys.argv) > 2 else list(range(10))
    sizes = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [64, 300]
    out.mkdir(parents=True, exist_ok=True)
    L.RES = out
    L.OUT = AL / "out"
    (L.OUT / "adapters").mkdir(parents=True, exist_ok=True)
    (L.OUT / "probs").mkdir(parents=True, exist_ok=True)
    tok = L.Enc(Path(L.stock_path()) / "tokenizer.json")
    L.PAD = tok.tok.token_to_id("[PAD]")
    lib = [x for x in WORKFLOWS if x != W]
    for x in lib:   # lora.py's "spike" branch, seed 0, LoRA kind
        f = L.OUT / "adapters" / f"S-lora-{x}-s0.pt"
        if f.exists():
            continue
        t0 = time.time()
        torch.manual_seed(0)
        cases = sorted(load_cases(x, "train"), key=lambda c: c["id"])
        random.Random(f"{x}-0").shuffle(cases)
        va, tr = L.prep(tok, cases[:30]), L.prep(tok, cases[30:])
        model = L.new_model("lora")
        info = L.train(model, "lora", tr, 10, 0, va=va[:4] if SMOKE else va, max_steps=SMOKE)
        torch.save(L.adapter_state(model), f)
        print(f"rebuilt S-lora-{x}-s0 in {time.time() - t0:.0f} s, best epoch {info['best_epoch']}", flush=True)
    test = L.prep(tok, load_cases(W, "test")[:4 if SMOKE else None])
    start = L.merged_adapter([torch.load(L.OUT / "adapters" / f"S-lora-{x}-s0.pt") for x in lib])
    for n in sizes:
        for seed in seeds:
            name = f"G-warm-{W}-n{n}-s{seed}"
            if (out / f"{name}.json").exists():
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            model = L.new_model("warm")
            L.load_adapter(model, start)
            tr = L.prep(tok, revealed(W, seed, n))
            ep = L.growth_epochs(n)
            info = L.train(model, "lora", tr, ep, seed, max_steps=SMOKE)
            L.evaluate(name, model, test, None, {**info, "kind": "warm", "workflow": W, "seed": seed, "n": n,
                                                 "epochs": ep, "sec": time.time() - t0})
            print(name, json.loads((out / f"{name}.json").read_text())["test"]["accuracy"], f"{time.time() - t0:.0f} s", flush=True)
    print("GROWTH_DONE", flush=True)


if __name__ == "__main__":
    main()
