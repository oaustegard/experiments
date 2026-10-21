"""Paired comparisons between shipped q4 builds: same queries, so the CI is on the
per-query difference (2,000 bootstrap resamples).

Usage: python3 -I run/pairs.py   (reads out/export/<tag>/model_q4.onnx; writes results/pairs.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import TESTS, Bench  # noqa: E402
from export import ort_embed  # noqa: E402
from model import ROOT, tokenize  # noqa: E402
from common import Teacher, overlap  # noqa: E402
from final_eval import paired  # noqa: E402

PAIRS = {
    "train_on_cut_vs_cut_after": ("l2_k4_ep2_v32k", "l2_k4_ep2"),
    "second_epoch_k4": ("l2_k4_ep2", "l2_k4"),
    "k4_two_epochs_vs_k8_one": ("l2_k4_ep2", "l2_k8"),
    "k8_vs_k4_one_epoch": ("l2_k8", "l2_k4"),
    "k4_vs_k2_one_epoch": ("l2_k4", "l2_k2"),
    "k2_second_epoch_on_cut": ("l2_k2_ep2_v32k", "l2_k2"),
    "mac_k4_10ep_vs_cpu_2ep": ("l2_mac_k4", "l2_k4_ep2_v32k"),
    "mac_k2_10ep_vs_cpu_2ep": ("l2_mac_k2", "l2_k2_ep2_v32k"),
    "mac_k8_vs_k4_10ep": ("l2_mac_k8", "l2_mac_k4"),
    "mac_k4_vs_k2_10ep": ("l2_mac_k4", "l2_mac_k2"),
}


def main():
    T, B = Teacher(), Bench()
    per = {}

    def get(tag):
        if tag not in per:
            d = ROOT / "out" / "export" / tag
            tok = Tokenizer.from_file(str(d / "tokenizer.json"))
            per[tag] = {}
            for t in TESTS:
                top = B.ix.topk(ort_embed(d / "model_q4.onnx", tokenize(T, B.texts(t), remap=tok)))
                per[tag][t] = overlap(top, B.top[t])
                if t == "synth_test":
                    per[tag]["hit"] = (top == B.src[:, None]).any(1).astype(float)
        return per[tag]

    res = {}
    for name, (x, y) in PAIRS.items():
        if not all((ROOT / "out" / "export" / t / "model_q4.onnx").exists() for t in (x, y)):
            continue
        a, b = get(x), get(y)
        res[name] = {"a": x, "b": y, **{t: paired(a[t], b[t]) for t in TESTS}, "hit@10": paired(a["hit"], b["hit"])}
        print(name, json.dumps({k: [round(v, 4) for v in res[name][k]] for k in (*TESTS, "hit@10")}), flush=True)
    (ROOT / "results" / "pairs.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
