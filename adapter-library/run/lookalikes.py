"""System v0.1.1: does the router flag inputs shaped like a workflow that belong to none of them?

    python3 -I run/lookalikes.py [router-v1 router-v2 ...]

Four synthetic look-alike domains, 25 inputs each, all of which should route to `none`:
  chat      a coworkers' chat thread in the customer-service `thread` shape (no customer, no account issue)
  weather   a weather alert in the security `alert` shape (description, evidence, source)
  receipt   a shop receipt in the invoice `invoice.lines` shape (no purchase order, no delivery, no terms)
  pipeline  a CI build in the agent-trace shape (task, steps, duration; no AI agent)
Expectation written before the first run: router-v1 flags fewer than half of them, because its `none`
cases were short texts and unrelated JSON. Also re-scores the in-library test cases and the unseen sets
at the shipped threshold (0.5). Writes results/lookalikes.json.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora  # noqa: E402
from lora import load_adapter, new_model, predict, prep_raw, stock_path  # noqa: E402
from shared import BD, ROOT, WORKFLOWS, load_cases  # noqa: E402
from system_router import SPIKES, TEST_DOMAIN, example, synthetic  # noqa: E402
from train_enc import Enc, probs_at  # noqa: E402

THR = 0.5


def lookalike(domain, i):
    r = random.Random(f"look-{domain}-{i}")
    if domain == "chat":
        lines = [("colleague", r.choice(["Lunch at the noodle place today?", "Who's bringing the projector to the offsite?",
                                         "Did anyone see the game last night?", "Can we move standup to 10?"])),
                 ("colleague", r.choice(["Sure, 12:30 works.", "I can grab it from the storeroom.", "Missed it, was it close?",
                                         "Fine by me, I'll update the invite."])),
                 ("colleague", r.choice(["Great, see you there.", "Thanks!", "Overtime, unbelievable.", "Done."]))]
        return {"channel": r.choice(["#team-lunch", "#offsite", "#random", "#standup"]),
                "thread": [{"role": a, "text": t} for a, t in lines[: r.randint(2, 3)]]}
    if domain == "weather":
        ev = r.choice(["River gauge at Millford rose 1.4 m in six hours; further rain expected overnight.",
                       "Wind gusts of 95 km/h recorded on the coast; ferries suspended until morning.",
                       "Temperatures above 38 C forecast for three consecutive days in the valley."])
        return {"alert": {"description": r.choice(["flood warning issued for the river basin", "severe wind warning",
                                                    "heat advisory for the region"]), "evidence": ev,
                          "source": "national weather service", "issued": f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"},
                "region": r.choice(["North Valley", "Coastal District", "Millford"]), "level": r.choice(["yellow", "amber", "red"])}
    if domain == "receipt":
        items = r.sample([("SKU-MILK", "milk 1L", 1.29), ("SKU-BREAD", "sourdough", 3.5), ("SKU-EGGS", "eggs x12", 4.1),
                          ("SKU-APPL", "apples 1kg", 2.4), ("SKU-COFF", "coffee beans", 9.9)], r.randint(2, 4))
        lines = [{"qty": (q := r.randint(1, 3)), "sku": s, "item": n, "unit_usd": u, "total_usd": round(q * u, 2)} for s, n, u in items]
        return {"invoice": {"id": f"RCPT-{r.randint(100000, 999999)}", "vendor": r.choice(["Corner Grocery", "Fresh Mart", "Daily Deli"]),
                            "lines": lines, "total_usd": round(sum(l["total_usd"] for l in lines), 2), "currency": "USD"},
                "payment": {"method": r.choice(["card", "cash"]), "status": "paid"}}
    steps = r.randint(4, 12)
    return {"pipeline": {"runner": "ci-runner-0" + str(r.randint(1, 9)), "trigger": r.choice(["push", "schedule", "merge request"])},
            "task": r.choice(["Build and test the web app.", "Publish the nightly docs site.", "Run the integration test suite."]),
            "trace_summary": {"steps": steps, "failed_steps": r.randint(0, 2), "duration_s": round(r.uniform(60, 1800), 1)},
            "artifacts": r.sample(["coverage.xml", "site.tar.gz", "junit.xml", "app.apk"], 2)}


CHAT = ["Lunch at the noodle place today?", "Sure, 12:30 works.", "Who's bringing the projector to the offsite?",
        "I can grab it from the storeroom.", "Did anyone see the game last night?", "Overtime, unbelievable."]
WEATHER = [("flood warning issued for the river basin", "River gauge at Millford rose 1.4 m in six hours; further rain expected overnight."),
           ("severe wind warning for the coast", "Wind gusts of 95 km/h recorded on the coast; ferries suspended until morning."),
           ("heat advisory for the valley", "Temperatures above 38 C forecast for three consecutive days in the valley.")]


def swapped(domain, cases):
    """Harder look-alikes (added after round one of this test flagged every shaped look-alike): a real test
    case's full structure with its content swapped for something outside the workflow."""
    out = []
    for i, c in enumerate(cases):
        st, r = json.loads(c["state"]), random.Random(f"swap-{domain}-{i}")
        if domain == "chat_in_support_shape":
            st["thread"] = [{"role": t["role"], "text": r.choice(CHAT)} for t in st["thread"]]
        else:
            d, e = r.choice(WEATHER)
            st["alert"] = {"description": d, "evidence": e, "rule": r.choice(["flood_watch", "wind_watch", "heat_watch"])}
        out.append((json.dumps(st, ensure_ascii=False), "none"))
    return out


def main():
    torch.set_num_threads(4)
    routers = sys.argv[1:] or ["router-v1"]
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    lora.PAD = tok.tok.token_to_id("[PAD]")
    texts = [json.loads(l)["input"] for l in open(BD / "data" / "decider_eval.jsonl")]
    random.Random("none-texts").shuffle(texts)
    sets = {"in_library": [(c["state"], w) for w in WORKFLOWS for c in load_cases(w, "test")],
            "unseen_texts": [(t, "none") for t in texts[70:]],
            "unseen_lab": [(json.dumps(synthetic(TEST_DOMAIN, i)), "none") for i in range(40)]}
    for d in ("chat", "weather", "receipt", "pipeline"):
        sets[f"lookalike_{d}"] = [(json.dumps(lookalike(d, i)), "none") for i in range(25)]
    sets["swapped_chat_in_support_shape"] = swapped("chat_in_support_shape", load_cases("customer_service", "test")[:25])
    sets["swapped_weather_in_alert_shape"] = swapped("weather_in_alert_shape", load_cases("security_incidents", "test")[:25])
    names = list(SPIKES) + ["none"]
    path = ROOT / "results" / "lookalikes.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for rn in routers:
        model = new_model("lora", qbase=rn in ("router-v2", "router-v3"))
        load_adapter(model, torch.load(ROOT / "out" / "adapters" / f"{rn}.pt"))
        model.eval()
        res = {}
        for k, v in sets.items():
            P = np.array([probs_at([z], 1.0)[0] for z in predict(model, prep_raw(tok, [example(s, l) for s, l in v]), lora.PAD)])
            route = [names[int(np.argmax(p[:-1]))] if p[-1] < THR else "none" for p in P]
            res[k] = {"correct": float(np.mean([r == l for r, (_, l) in zip(route, v)])), "flagged_none": float(np.mean([r == "none" for r in route])),
                      "routed_to": {n: route.count(n) for n in names if route.count(n)}, "median_p_none": float(np.median(P[:, -1]))}
            print(rn, k, res[k], flush=True)
        res["lookalike_mean_flagged"] = float(np.mean([res[k]["flagged_none"] for k in res if k.startswith("lookalike_")]))
        res["swapped_mean_flagged"] = float(np.mean([res[k]["flagged_none"] for k in res if k.startswith("swapped_")]))
        out[rn] = res
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
