"""Per-type accuracy and score-within-one-level for the 10-epoch stock specialists, from saved predictions.

Score +-1 is the column nanoGentzen-Jev-v3's card reports; the dataset scorer does not compute it.
"""
import json
from pathlib import Path

from common import examples, load_cases

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
W = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
key = {(e["case"], e["qname"]): e for e in examples([c for w in W for c in load_cases(w, "test")])}
out = {"values": [], "card_nanoGentzen_Jev_v3": {"overall": 0.731, "noul": 0.828, "choice": 0.685, "score": 0.693,
       "score_pm1": 0.973, "kl": 0.150, "brier": 0.081, "ece": 0.087}}
for s in (0, 1, 2):
    within = n = 0
    for r in json.loads((RES / f"spec-stock-e10-s{s}_preds.json").read_text()):
        e = key[(r["case"], r["q"])]
        if e["type"] != "score":
            continue
        ap, ag = r["p"].index(max(r["p"])), e["gold"].index(max(e["gold"]))
        within += abs(int(e["options"][ap][0]) - int(e["options"][ag][0])) <= 1
        n += 1
    out["values"].append(round(within / n, 4))
out["mean"] = round(sum(out["values"]) / 3, 4)
(RES / "score_pm1.json").write_text(json.dumps(out, indent=1) + "\n")
print(out)
