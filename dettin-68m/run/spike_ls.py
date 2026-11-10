"""LoRA r8 spikes on the frozen stock base for the life-science suite: one adapter per question group.

    AL_BASE=jhu-clsp/ettin-encoder-68m AL_AMP=1 python3 run/spike_ls.py CASES.jsonl OUT_DIR [--seeds 0 1]

Uses adapter-library's LoRA module and training loop unchanged (rank 8 on Wqkv, both Wo and Wi, lr 5e-4).
For each group in suite.SPIKE, trains on the cases' questions in that group and scores the test cases.
Writes OUT_DIR/spike-<group>-s<seed>.json (summary), ..._preds.json and adapters/<name>.pt.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "adapter-library" / "run"))
import lora
from common import examples, score_examples
from suite import SPIKE
from train_enc import (
    Enc,
    fit_temps,
    predict,
    probs_at,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    with open(a.cases) as f:
        recs = [json.loads(line) for line in f]
    tok = Enc(Path(lora.stock_path()) / "tokenizer.json")
    lora.PAD = tok.tok.token_to_id("[PAD]")

    def prep(cases, names):
        sub = []
        for c in cases:
            qs = {k: v for k, v in c["questions"].items() if k in names}
            if qs:
                sub.append({**c, "questions": qs})
        res = []
        for e in examples(sub):
            ids, pos = tok.build(e)
            res.append({**e, "ids": ids, "pos": pos})
        return res

    for group, names in SPIKE.items():
        for seed in a.seeds:
            name = f"spike-{group}-s{seed}"
            if (out / f"{name}.json").exists():
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            train = [c for c in recs if c["split"] == "train"]
            accts = sorted({c["account"] for c in train})
            random.Random(seed).shuffle(accts)
            val_acc = set(accts[: max(5, len(accts) // 10)])
            tr = prep([c for c in train if c["account"] not in val_acc], names)
            va = prep([c for c in train if c["account"] in val_acc], names)
            te = prep([c for c in recs if c["split"] == "test"], names)
            model = lora.new_model("lora")
            info = lora.train(model, "lora", tr, a.epochs, seed, va=va, max_steps=a.max_steps)
            info["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            temps = fit_temps(va, predict(model, va, lora.PAD))
            lg = predict(model, te, lora.PAD)
            probs = [probs_at([z], temps.get(d["type"], 1.0))[0] for z, d in zip(lg, te)]
            summ, rows = score_examples(te, probs)
            torch.save(lora.adapter_state(model), out / "adapters" / f"{name}.pt")
            (out / f"{name}.json").write_text(json.dumps({"name": name, "group": group, "test": summ, "temps": temps,
                                                          **info, "n_train": len(tr), "sec": time.time() - t0}, indent=1))
            (out / f"{name}_preds.json").write_text(json.dumps(
                [{"case": d["case"], "q": d["qname"], "p": [round(x, 5) for x in p], "ok": r[0]}
                 for d, p, r in zip(te, probs, rows)]))
            print(f"[{name}] acc {summ['accuracy']:.4f} trainable {info['trainable']} {time.time() - t0:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()
    print("SPIKES_LS_DONE", flush=True)


if __name__ == "__main__":
    main()
