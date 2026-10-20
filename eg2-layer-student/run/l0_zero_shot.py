"""L0: EG2 cut to its first k layers, no training, for each k.

Usage: python3 -I run/l0_zero_shot.py [--ks 1 2 3 4 6 8 12 16 20 24]

Weights are the q4 teacher's, dequantized (run/model.py), so k = 24 is EG2 and every
loss is the cut's. Writes results/l0.json (per k: params by part, and the bench scores).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import TESTS, Bench  # noqa: E402
from model import ROOT, Student, embed, tokenize  # noqa: E402
from common import Teacher  # noqa: E402


def params(m):
    parts = {"embed_tokens": 0, "layers": 0, "ple_projection": 0, "head": 0}
    for n, p in m.tm.named_parameters():
        key = "embed_tokens" if n.startswith("embed_tokens") else "layers" if n.startswith("layers") \
            else "ple_projection" if n.startswith("ple") else "head"
        parts[key] += p.numel()
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8, 12, 16, 20, 24])
    a = ap.parse_args()
    torch.set_num_threads(4)
    T, B = Teacher(), Bench()
    enc = {t: tokenize(T, B.texts(t)) for t in TESTS}
    out = ROOT / "results" / "l0.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for k in a.ks:
        if str(k) in res:
            continue
        t0 = time.time()
        m = Student(k)
        P = {t: embed(m, enc[t]) for t in TESTS}
        r, _ = B.score(P)
        res[str(k)] = {"params": params(m), **r, "seconds": time.time() - t0}
        print(k, json.dumps({t: r[t]["overlap@10"][0] for t in TESTS}),
              "hit", r["synth_test"]["hit@10"][0], flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
