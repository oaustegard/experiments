"""System v0.1.0: train the router spike over the four whole-workflow spikes plus `none`.

    python3 -I run/system_router.py [--smoke]

Options are the spikes' descriptions (SPIKES below, also shipped to the page) plus `none`. Training
cases: every Typed Decisions train case, labelled with its workflow; `none` cases are out-of-domain
inputs: 70 of the 104 short texts in browser-decider's decider_eval.jsonl, plus synthetic JSON states
from three unrelated domains (staff leave, restaurant bookings, property listings). A fourth synthetic
domain (lab samples) and the other 34 texts are never trained on: they test gap detection on kinds of
input the router has not seen. The `none` threshold is set on validation so at most 1% of in-library
cases are flagged. Writes out/adapters/router-v1.pt and results/system_router.json.
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
from lora import adapter_state, new_model, predict, prep_raw, stock_path, train  # noqa: E402
import lora  # noqa: E402
from shared import BD, ROOT, WORKFLOWS, load_cases  # noqa: E402
from train_enc import Enc, probs_at  # noqa: E402

SPIKES = {
    "customer_service": "Customer support conversation: a ticket thread with the customer's account and orders. "
                        "Decides the category, the next action, whether a human is needed, urgency and churn risk.",
    "invoice_processing": "Vendor invoice review: an invoice with its purchase order, delivery record and payment terms. "
                          "Decides the disposition, discrepancy severity, duplicates, the order match and urgency.",
    "security_incidents": "Security alert triage: an alert with its evidence, the principal involved and context. "
                          "Decides the disposition, severity, whether it is a true positive or a credential compromise, and urgency.",
    "agent_trace_observability": "AI agent run trace: an autonomous agent's task, constraints and step summary. "
                                 "Decides the outcome, risk, whether it needs review, urgency and what observability should do.",
}
NONE_DESC = "None of these specialists covers this input."
QUESTION = "choice: Which specialist should handle this input? Pick the one whose description fits it."
TRAIN_DOMAINS, TEST_DOMAIN = ["leave", "booking", "listing"], "lab"


def synthetic(domain, i):
    r = random.Random(f"{domain}-{i}")
    if domain == "leave":
        return {"employee": {"id": f"E-{r.randint(1000, 9999)}", "department": r.choice(["finance", "design", "ops", "legal"]),
                             "tenure_years": r.randint(0, 20)},
                "request": {"type": r.choice(["annual", "sick", "parental", "unpaid"]), "days": r.randint(1, 30),
                            "start": f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"},
                "note": r.choice(["Family trip planned months ago.", "Doctor's note attached.", "Covering arranged with the team.",
                                  "Short notice, sorry."])}
    if domain == "booking":
        return {"restaurant": r.choice(["Harbour Grill", "Osteria Nove", "Saffron House"]),
                "booking": {"party_size": r.randint(1, 12), "time": f"{r.randint(17, 22)}:{r.choice(['00', '30'])}",
                            "date": f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"},
                "requests": r.sample(["window seat", "high chair", "gluten free", "quiet table", "birthday cake"], r.randint(0, 2))}
    if domain == "listing":
        return {"property": {"type": r.choice(["flat", "house", "studio"]), "bedrooms": r.randint(0, 5),
                             "area_m2": r.randint(25, 240), "city": r.choice(["Leeds", "Porto", "Graz"])},
                "price_eur": r.randint(80, 900) * 1000, "agent_notes": r.choice(["Needs a new roof.", "Recently renovated.",
                                                                                "South-facing garden.", "Close to the station."])}
    return {"sample": {"id": f"LS-{r.randint(10000, 99999)}", "matrix": r.choice(["serum", "soil", "water", "tissue"]),
                       "collected": f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"},
            "assays": [{"name": r.choice(["pH", "lead", "glucose", "nitrate"]), "value": round(r.uniform(0, 50), 2),
                        "unit": r.choice(["mg/L", "ppm", "mmol/L"])} for _ in range(r.randint(1, 3))],
            "chain_of_custody": r.choice(["intact", "broken seal", "late delivery to lab"])}


TRAIN_CHAT = ["Has everyone finished the book for Thursday?", "I'm halfway, the ending better be good.",
              "Anyone have a good lentil soup recipe?", "Add cumin early, it changes everything.",
              "Book club moved to the café on Elm Street.", "Perfect, see you all there."]
TRAIN_NOTICES = [("school closed for teacher training day", "All classes are cancelled on Friday; the canteen and buses will not run."),
                 ("bake sale this weekend", "Parents are asked to bring labelled trays to the hall by 9 am on Saturday."),
                 ("library returns overdue", "Three books checked out in September are now two weeks late; fines start Monday.")]


def swap_negatives(n=40):
    """`none` cases in the support and alert shapes, built from train cases (the test look-alikes use
    lunch-chat and weather content, run/lookalikes.py)."""
    out = []
    for w, field in (("customer_service", "thread"), ("security_incidents", "alert")):
        for i, c in enumerate(load_cases(w, "train")[:n]):
            st, r = json.loads(c["state"]), random.Random(f"trainswap-{w}-{i}")
            if field == "thread":
                st["thread"] = [{"role": t["role"], "text": r.choice(TRAIN_CHAT)} for t in st["thread"]]
            else:
                d, e = r.choice(TRAIN_NOTICES)
                st["alert"] = {"description": d, "evidence": e, "rule": r.choice(["closure_notice", "event_notice", "library_notice"])}
            out.append((json.dumps(st, ensure_ascii=False), "none"))
    return out


def example(state, label):
    names = list(SPIKES) + ["none"]
    return {"case": "", "workflow": "", "qname": "router", "type": "choice", "question": QUESTION, "instructions": QUESTION,
            "options": [(k, v) for k, v in SPIKES.items()] + [("none", NONE_DESC)], "state": state,
            "gold": [1.0 if n == label else 0.0 for n in names]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--qbase", action="store_true", help="v0.1.1: train on the quantized base; writes router-v2")
    ap.add_argument("--swap-negatives", action="store_true",
                    help="v0.1.1: add `none` cases that keep a real support or alert state's structure with off-topic "
                         "content (book club and recipe chat, school notices); writes router-v3")
    a = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    lora.PAD = tok.tok.token_to_id("[PAD]")
    texts = [json.loads(l)["input"] for l in open(BD / "data" / "decider_eval.jsonl")]
    random.Random("none-texts").shuffle(texts)
    tr_none = [(t, "none") for t in texts[:70]] + [(json.dumps(synthetic(d, i)), "none") for d in TRAIN_DOMAINS for i in range(40)]
    te_none_text = [(t, "none") for t in texts[70:]]
    te_none_lab = [(json.dumps(synthetic(TEST_DOMAIN, i)), "none") for i in range(40)]
    if a.swap_negatives:
        tr_none += swap_negatives()
    tr = [(c["state"], w) for w in WORKFLOWS for c in load_cases(w, "train")] + tr_none
    random.Random("router-v1").shuffle(tr)
    if a.smoke:
        tr = tr[:48]
    ex = prep_raw(tok, [example(s, l) for s, l in tr])
    nval = max(20, len(ex) // 10)
    va, trn = ex[:nval], ex[nval:]
    torch.manual_seed(0)
    model = new_model("lora", qbase=a.qbase)
    info = train(model, "lora", trn, 10, 0, va=va, max_steps=2 if a.smoke else 0)
    names = list(SPIKES) + ["none"]

    def run(items):
        e = prep_raw(tok, [example(s, l) for s, l in items])
        return np.array([probs_at([z], 1.0)[0] for z in predict(model, e, lora.PAD)])

    pv = np.array([probs_at([z], 1.0)[0] for z in predict(model, va, lora.PAD)])
    lab_v = [names[int(np.argmax(e["gold"]))] for e in va]
    in_lib = np.array([l != "none" for l in lab_v])
    thr = float(np.quantile(pv[in_lib, -1], 0.99)) if in_lib.any() else 0.5
    te = [(c["state"], w) for w in WORKFLOWS for c in load_cases(w, "test")][: 8 if a.smoke else None]
    pt, pn_text, pn_lab = run(te), run(te_none_text), run(te_none_lab)
    route = lambda p: names[int(np.argmax(p[:-1]))] if p[-1] < thr else "none"
    res = {"spikes": SPIKES, "none_desc": NONE_DESC, "question": QUESTION, "threshold_p_none": thr, **info,
           "test_routing_acc": float(np.mean([route(p) == l for p, (_, l) in zip(pt, te)])),
           "test_false_gap_rate": float(np.mean([route(p) == "none" for p in pt])),
           "gap_recall_unseen_texts": float(np.mean([route(p) == "none" for p in pn_text])),
           "gap_recall_unseen_domain_lab": float(np.mean([route(p) == "none" for p in pn_lab])),
           "p_none_median": {"in_library_test": float(np.median(pt[:, -1])), "unseen_texts": float(np.median(pn_text[:, -1])),
                             "unseen_lab_states": float(np.median(pn_lab[:, -1]))},
           "sec": time.time() - t0}
    out = ROOT / ("out/smoke" if a.smoke else "out") / "adapters"
    out.mkdir(parents=True, exist_ok=True)
    tag = "v3" if a.swap_negatives else "v2" if a.qbase else "v1"
    torch.save(adapter_state(model), out / f"router-{tag}.pt")
    dst = ROOT / ("out/smoke" if a.smoke else "results") / {"v1": "system_router.json", "v2": "system_router_q.json",
                                                              "v3": "system_router_v3.json"}[tag]
    dst.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("spikes", "log")}, indent=1))


if __name__ == "__main__":
    main()
