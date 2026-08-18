"""Does STE-style text make Baguettotron more predictable?

Acceptance in speculative decoding is bounded by how peaked the target's own
next-token distribution is. A controlled language constrains syntax and word
choice, so the question is whether that shows up as lower entropy under the
target model. If it does, the gain from ASD-STE100 comes from the domain the
model is decoding in, not from the drafter's vocabulary.

Corpora are hand-written proxies: the STE arm follows the published rules
(short imperative sentences, one instruction per sentence, no -ing forms, no
passive in procedures, approved-style general vocabulary plus technical names),
but the official dictionary is copyrighted and gated, so this is STE-shaped
prose rather than certified STE.
"""
import json, math, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
model = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron", dtype=torch.float32).eval()

STE = """
Remove the access panel. Loosen the four bolts. Do not remove the bolts fully.
Put the panel on a clean surface. Examine the seal for damage. If the seal has
damage, replace the seal. Obey the torque values in Table 3. Tighten the bolts
to 25 Nm. Do a check of the hydraulic pressure. If the pressure is less than
2000 psi, do the procedure again. Make sure that the valve is closed before you
start. Warning: hydraulic fluid is an irritant. Wear gloves and eye protection.
Disconnect the electrical power before you touch the connector. Put a cap on the
open port. Record the results in the log. Install the access panel. Tighten the
four bolts to 25 Nm. Do a leak check. If you find a leak, tell the supervisor.
Clean the area. Remove the tools from the work area. Close the log entry.
"""

GENERAL = """
The Roman aqueducts were among the most ambitious engineering projects of the
ancient world, and their construction demanded a sustained gradient across
terrain that rarely cooperated. Surveyors used the chorobates, a levelling
instrument whose accuracy has been debated ever since, to hold a fall of a few
centimetres per kilometre over distances that sometimes exceeded fifty. Where
valleys interrupted the line, engineers had a choice between an arcade and an
inverted siphon, and the decision turned as much on the availability of lead and
labour as on hydraulics. Maintenance was continuous rather than occasional,
since calcium carbonate accumulated on the channel walls and slowly strangled
the flow.
"""

CODE = """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""


def profile(name, text):
    ids = tok(text.strip(), add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    p = lp.exp()
    ent = -(p * lp).sum(-1)                      # nats, per position
    top1 = p.max(-1).values
    tgt = ids[0, 1:]
    tgt_p = p.gather(1, tgt.unsqueeze(1)).squeeze(1)
    # Renyi-2 collision probability doubles as an acceptance proxy: two
    # independent draws from the same distribution agree with probability
    # sum(p^2), which is the ceiling for a drafter that has learned the target
    # exactly and then samples rather than copies.
    coll = (p ** 2).sum(-1)
    return {
        "corpus": name,
        "tokens": int(ids.shape[1]),
        "mean_entropy_nats": round(float(ent.mean()), 3),
        "median_entropy_nats": round(float(ent.median()), 3),
        "mean_top1_prob": round(float(top1.mean()), 4),
        "median_top1_prob": round(float(top1.median()), 4),
        "mean_gold_token_prob": round(float(tgt_p.mean()), 4),
        "greedy_hit_rate": round(float((p.argmax(-1) == tgt).float().mean()), 4),
        "mean_collision_prob": round(float(coll.mean()), 4),
        "effective_vocab_exp_entropy": round(float(ent.mean().exp()), 1),
    }


rows = [profile("ste_style_procedure", STE),
        profile("general_prose", GENERAL),
        profile("code", CODE)]
for r in rows:
    print(json.dumps(r), flush=True)

base = next(r for r in rows if r["corpus"] == "general_prose")
out = {"rows": rows,
       "ste_vs_general": {
           "top1_prob_ratio": round(rows[0]["mean_top1_prob"] / base["mean_top1_prob"], 3),
           "entropy_delta_nats": round(rows[0]["mean_entropy_nats"]
                                       - base["mean_entropy_nats"], 3),
           "greedy_hit_ratio": round(rows[0]["greedy_hit_rate"]
                                     / base["greedy_hit_rate"], 3),
       }}
print(json.dumps(out["ste_vs_general"], indent=2))
json.dump(out, open("ste_entropy.json", "w"), indent=2)
