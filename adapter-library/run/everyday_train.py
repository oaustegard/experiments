"""System v0.2.0: train the everyday spikes (transactions, emails, reviews) and their router. GPU.

    python3 -I run/everyday_train.py [--smoke]

Spikes follow v0.1.1's winning recipe: 10 epochs of LoRA on the fp32 base, then 5 epochs on the
fake-quantized base that ships, starting from that adapter (validation KL picks the epoch, per-type
temperature on validation). The router is a LoRA decision model on the quantized base: one choice
question, the three spike descriptions plus `none`. `none` trains on 70 decider_eval texts, synthetic JSON
from three unrelated domains and off-topic texts of six kinds (recipes, poems, news, posts, fiction,
meeting notes); it is tested on the other 34 texts, synthetic lab records and five unseen kinds (code,
sports results, weather, lyrics, product specs). Writes out/adapters/E-*.pt, results/E-*.json and
out/probs/E-*.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora  # noqa: E402
from lora import adapter_state, load_adapter, new_model, predict, prep_raw, stock_path, train  # noqa: E402
from shared import BD, ROOT, examples, score_examples  # noqa: E402
from system_router import synthetic  # noqa: E402
from train_enc import Enc, fit_temps, probs_at  # noqa: E402

DOMAINS = ["transactions", "emails", "reviews"]
SPIKES = {
    "transactions": "Bank or card transaction: a merchant descriptor, amount and date, with recent charges from the same merchant. "
                    "Decides the spending category, whether it recurs and whether it is unusual.",
    "emails": "Email: a subject line and a body. Decides what kind of email it is, whether it needs a reply and how urgent it is.",
    "reviews": "Customer review of a business or product. Decides the star rating, the main complaint and whether the business should reply.",
}
NONE_DESC = "None of these specialists covers this input."
QUESTION = "choice: Which specialist should handle this input? Pick the one whose description fits it."
DATA = ROOT / "data" / "everyday"
RES, OUT = ROOT / "results", ROOT / "out"


def router_example(state, label):
    names = list(SPIKES) + ["none"]
    return {"case": "", "workflow": "", "qname": "router", "type": "choice", "question": QUESTION, "instructions": QUESTION,
            "options": [(k, v) for k, v in SPIKES.items()] + [("none", NONE_DESC)], "state": state,
            "gold": [1.0 if n == label else 0.0 for n in names]}


def none_sets():
    texts = [json.loads(l)["input"] for l in open(BD / "data" / "decider_eval.jsonl")]
    random.Random("none-texts").shuffle(texts)
    off = [json.loads(l) for l in open(DATA / "offtopic.jsonl")]
    train = ([(t, "none") for t in texts[:70]] + [(json.dumps(synthetic(d, i)), "none") for d in ("leave", "booking", "listing") for i in range(40)]
             + [(o["text"], "none") for o in off if o["split"] == "train"])
    test = {"unseen_texts": [(t, "none") for t in texts[70:]],
            "unseen_lab_records": [(json.dumps(synthetic("lab", i)), "none") for i in range(40)],
            "unseen_kinds": [(o["text"], "none") for o in off if o["split"] == "test"]}
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    lora.PAD = tok.tok.token_to_id("[PAD]")
    RES.mkdir(exist_ok=True)
    (OUT / "adapters").mkdir(parents=True, exist_ok=True)
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(DATA / "cases.jsonl")]
    ms = 2 if a.smoke else 0
    prep = lambda cs: prep_raw(tok, examples(cs))

    for d in DOMAINS:
        t0 = time.time()
        tr_all = sorted([c for c in cases if c["workflow"] == d and c["split"] == "train"], key=lambda c: c["id"])
        random.Random(f"everyday-{d}").shuffle(tr_all)
        te = [c for c in cases if c["workflow"] == d and c["split"] == "test"]
        if a.smoke:
            tr_all, te = tr_all[:24], te[:4]
        nval = max(4, len(tr_all) // 10)
        va, tr, tex = prep(tr_all[:nval]), prep(tr_all[nval:]), prep(te)
        torch.manual_seed(0)
        m1 = new_model("lora")
        i1 = train(m1, "lora", tr, 10, 0, va=va, max_steps=ms)
        st1 = adapter_state(m1)
        torch.manual_seed(0)
        m2 = new_model("lora", qbase=True)
        load_adapter(m2, st1)
        pre = score_examples(tex, probs_at(predict(m2, tex, lora.PAD), 1.0))[0]["accuracy"]   # stage-1 adapter on the shipped base
        i2 = train(m2, "lora", tr, 5, 0, va=va, max_steps=ms)
        temps = fit_temps(va, predict(m2, va, lora.PAD))
        lg = predict(m2, tex, lora.PAD)
        probs = [probs_at([z], temps.get(e["type"], 1.0))[0] for z, e in zip(lg, tex)]
        summ, _ = score_examples(tex, probs)
        by_q = {q: score_examples([e for e in tex if e["qname"] == q], [p for p, e in zip(probs, tex) if e["qname"] == q])[0]["accuracy"]
                for q in sorted({e["qname"] for e in tex})}
        torch.save(adapter_state(m2), OUT / "adapters" / f"E-{d}.pt")
        (OUT / "probs" / f"E-{d}.json").write_text(json.dumps([[round(x, 5) for x in p] for p in probs]))
        res = {"name": f"E-{d}", "domain": d, "n_train": len(tr_all) - nval, "n_val": nval, "n_test": len(te), "test": summ,
               "by_question": by_q, "stage1_on_quantized_base_acc": pre, "temps": temps, "stage1": i1, "stage2": i2, "sec": time.time() - t0}
        (RES / f"E-{d}.json").write_text(json.dumps(res, indent=1))
        print(f"[E-{d}] test acc {summ['accuracy']:.4f} {by_q} (stage 1 on quantized base {pre:.4f}) {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    tr_none, te_none = none_sets()
    tr = [(c["state"], c["workflow"]) for c in cases if c["split"] == "train"] + tr_none
    random.Random("everyday-router").shuffle(tr)
    if a.smoke:
        tr = tr[:48]
    ex = prep_raw(tok, [router_example(s, l) for s, l in tr])
    nval = max(20, len(ex) // 10)
    torch.manual_seed(0)
    model = new_model("lora", qbase=True)
    info = train(model, "lora", ex[nval:], 10, 0, va=ex[:nval], max_steps=ms)
    names = list(SPIKES) + ["none"]
    sets = {"in_library": [(c["state"], c["workflow"]) for c in cases if c["split"] == "test"][: 8 if a.smoke else None], **te_none}
    res = {"name": "E-router", "spikes": SPIKES, "none_desc": NONE_DESC, "question": QUESTION, "threshold_p_none": 0.5, **info}
    for k, v in sets.items():
        P = np.array([probs_at([z], 1.0)[0] for z in predict(model, prep_raw(tok, [router_example(s, l) for s, l in v]), lora.PAD)])
        route = [names[int(np.argmax(p[:-1]))] if p[-1] < 0.5 else "none" for p in P]
        res[k] = {"correct": float(np.mean([r == l for r, (_, l) in zip(route, v)])), "flagged_none": float(np.mean([r == "none" for r in route])),
                  "n": len(v), "routed_to": {n: route.count(n) for n in names if route.count(n)}}
        print("[E-router]", k, res[k], flush=True)
    res["sec"] = time.time() - t0
    torch.save(adapter_state(model), OUT / "adapters" / "E-router.pt")
    (RES / "E-router.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
