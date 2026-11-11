"""The zero-shot benchmark through GRef on a packed model (same metric, thresholds and JSON as bench.py).

    python3 run/gbench_packed.py --model web/models/<name>-<variant> --out results/<name>.json
                                 [--limit N] [--splits dev,test] [--sets ...] [--logit-bias B]
                                 [--verify-ckpt CKPT_DIR]

bench.py's own run_bench drives the run (same sets, splits, thresholds, low-threshold decode, prf and
aggregation, so the JSON has the same shape); only its per-batch `predict` is replaced by one that runs
GRef sentence by sentence on the benchmark's pre-split words: one forward per sentence, then the
library's greedy decode at the low threshold and at each reported threshold from the same logits.

--logit-bias adds to the manifest's logit bias (a test hook: barely trained checkpoints predict nothing).
--verify-ckpt also runs bench.py's library path on the PyTorch checkpoint with the same limit, sets and
bias, and records whether every per-set, per-split, per-threshold count (tp, n_pred, n_true) and F1 is
identical.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "data" / "hf"))
import bench  # noqa: E402
import gref  # noqa: E402


class GModel:
    """The few attributes bench.run_bench reads from a model, around a GRef."""

    def __init__(self, g):
        self.g, self.training, self.device = g, False, "numpy (GRef)"
        self.n_sent, self.seconds = 0, 0.0

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def gref_predict(model, data, labels, thresholds=bench.THRESHOLDS, low=bench.LOW, batch_size=12):
    g = model.g
    trues, low_preds, preds = [], [], {t: [] for t in thresholds}
    t0 = time.perf_counter()
    for r in data:
        out = g.predict_words(r["tokenized_text"], labels, thresholds=(low, *thresholds))
        low_preds.append([(s, e, lab, sc) for s, e, lab, sc in out[low]])
        for t in thresholds:
            preds[t].append([(s, e, lab) for s, e, lab, _ in out[t]])
        trues.append(list(r["ner"]))
    model.seconds += time.perf_counter() - t0
    model.n_sent += len(data)
    return trues, low_preds, preds


def flat(res):
    """{(set, split, threshold): (tp, n_pred, n_true, f1)} from a run_bench result."""
    out = {}
    for s, splits in res["sets"].items():
        for sp, r in splits.items():
            for t, v in r["by_threshold"].items():
                out[(s, sp, t)] = (v["tp"], v["n_pred"], v["n_true"], v["f1"])
            lt = r["low_threshold"]
            out[(s, sp, "low")] = (lt["tp"], lt["n_pred"], lt["n_true"], lt["f1"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path, help="packed model dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--logit-bias", type=float, default=0.0)
    ap.add_argument("--verify-ckpt", default=None, help="checkpoint dir or cached hub id")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    g = gref.load(a.model)
    g.b += a.logit_bias
    gm = GModel(g)
    bench.predict = gref_predict            # run_bench -> bench_split -> predict (module global)
    splits = tuple(a.splits.split(","))
    sets = a.sets.split(",") if a.sets else None
    res = bench.run_bench(gm, splits=splits, sets=sets, limit=a.limit)
    sps = gm.n_sent / gm.seconds if gm.seconds else None
    out = {"name": a.out.stem, "model": str(a.model), "class": f"{type(g).__name__}({g.g['span_mode']})",
           "variant": g.m["variant"], "head_format": g.m["head_format"], "prefix_space": g.g["prefix_space"],
           "logit_bias_extra": a.logit_bias, "versions": bench.versions(), "wall_seconds": round(time.time() - t0, 1),
           "gref_sentences": gm.n_sent, "gref_seconds": round(gm.seconds, 2),
           "gref_sentences_per_second": round(sps, 1) if sps else None, "bench": res}
    print(f"GRef: {gm.n_sent} sentences in {gm.seconds:.1f}s = {sps:.1f} sentences/s (1 process, numpy)")
    if a.verify_ckpt:
        import importlib
        import torch
        torch.set_grad_enabled(False)
        bench_lib = importlib.reload(bench)    # restore the library predict
        model = bench_lib.load_model(a.verify_ckpt, "cpu")
        model.config.logit_bias = (model.config.logit_bias or 0.0) + a.logit_bias
        t1 = time.time()
        lib = bench_lib.run_bench(model, splits=splits, sets=sets, limit=a.limit, log=None)
        fl, fg = flat(lib), flat(res)
        diffs = [{"key": list(k), "library": fl[k], "gref": fg.get(k)} for k in fl if fl[k] != fg.get(k)]
        out["verify"] = {"checkpoint": str(a.verify_ckpt), "cells": len(fl), "cells_identical": len(fl) - len(diffs),
                         "identical": not diffs and set(fl) == set(fg), "differences": diffs[:20],
                         "library_seconds": round(time.time() - t1, 1),
                         "library_test_mean7": lib.get("test_mean7"), "library_dev_mean5": lib.get("dev_mean5")}
        print(f"verify vs library: {len(fl) - len(diffs)}/{len(fl)} (set, split, threshold) cells identical")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for key in ("test_mean7", "dev_mean5"):
        if key in res:
            print(key, " ".join(f"{k}:{v * 100:.2f}" for k, v in res[key].items()))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
