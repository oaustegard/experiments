"""Score a trained decider on the hub's decider_eval.jsonl (schemas it never saw).

    python3 -I run/eval_ood.py CKPT_NAME [--eval PATH]

104 items: 64 banking77 card intents (8-way choice), 24 agent-guardrail predicates (yes/no),
16 rubric scores. Jev scores 0.894 and Haiku `decider` 0.846 on the same file (hub docs/decider.md).
Predicates map to `noul` with the default true/false options; score levels to options "0".."n-1".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NOUL_DEFAULT, ROOT, score_examples  # noqa: E402
from train_enc import Decider, Enc, predict, probs_at  # noqa: E402

# copied verbatim from claude-workspace tests/data/decider_eval.jsonl (2026-11-05)
HUB_EVAL = ROOT / "data" / "decider_eval.jsonl"


def convert(row):
    q = row["questions"][0]
    lab = row["labels"][q["name"]]
    if q["type"] == "choice":
        opts = [(c["value"], c["description"]) for c in q["choices"]]
        gi = [o for o, _ in opts].index(lab)
        typ = "choice"
    elif q["type"] == "predicate":
        opts = list(NOUL_DEFAULT.items())
        gi = 0 if lab else 1
        typ = "noul"
    else:
        opts = [(str(i), f"{lv['label']}: {lv['description']}") for i, lv in enumerate(q["levels"])]
        gi = int(lab)
        typ = "score"
    gold = [0.0] * len(opts)
    gold[gi] = 1.0
    return {"case": row["id"], "workflow": row["source"].split(" ")[0], "qname": q["name"], "type": typ,
            "question": f"{typ}: {q['instructions']}", "options": opts, "state": row["input"], "gold": gold,
            "src_type": q["type"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--eval", default=str(HUB_EVAL))
    a = ap.parse_args()
    torch.set_num_threads(2)
    ck = ROOT / "out" / "ckpt" / a.name
    meta = json.loads((ck / "meta.json").read_text())
    tok = Enc(Path(meta["init"]) / "tokenizer.json")
    rows = [json.loads(l) for l in open(a.eval)]
    ex = [convert(r) for r in rows]
    for e in ex:
        e["ids"], e["pos"] = tok.build(e)
    model = Decider(meta["init"])
    model.load_state_dict(torch.load(ck / "best.pt", map_location="cpu"))
    logits = predict(model, ex, tok.tok.token_to_id("[PAD]"))
    probs = [probs_at([z], meta["temps"][e["type"]])[0] for z, e in zip(logits, ex)]
    summ, rows_ = score_examples(ex, probs)
    summ.pop("kl_from_gold")  # gold is one-hot here; KL is just -log p(label)
    by = {}
    for t in ("choice", "predicate", "score"):
        sel = [i for i, e in enumerate(ex) if e["src_type"] == t]
        by[t] = {"n": len(sel), "accuracy": float(np.mean([rows_[i][0] for i in sel]))}
    # predicate base rate, to read the yes/no number against
    preds = [r for r in rows if r["questions"][0]["type"] == "predicate"]
    by["predicate"]["share_true"] = float(np.mean([r["labels"][r["questions"][0]["name"]] for r in preds]))
    res = {"name": a.name, "eval": "decider_eval.jsonl", "summary": summ, "by_type": by,
           "reference": {"jev": 0.894, "haiku_decider": 0.846}}
    (ROOT / "results" / f"ood_{a.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
