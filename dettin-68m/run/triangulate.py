"""Answer a held-out workflow's questions with the other workflows' spikes, combined by distance.

    python3 run/triangulate.py      # -> results/triangulate.json

Typed Decisions has four workflows and one 68M spike for each (results/a, seed 0). For each workflow in turn
its own spike is set aside and the other three answer its 100 test cases (500 questions). A case's distance to
a spike is the cosine between the case's state, mean-pooled through the frozen packed base with no spike
loaded, and the centroid of 60 training states of that spike's workflow. Combinations of the three
probability vectors per question: uniform mean, distance-weighted mean (softmax of cosine / TAU), the nearest
spike alone, and for reference each spike alone and the best single spike chosen after the fact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "browser-decider" / "run"))
from common import WORKFLOWS, load_cases  # noqa: E402
from dettin import Dettin  # noqa: E402

ROOT = HERE.parent
TAU = 0.02
N_TEST, N_CENT = 100, 60


@torch.no_grad()
def state_vec(m, text):
    ids = [m.ids["[CLS]"]] + m.tok.encode(text, add_special_tokens=False).ids[:1000] + [m.ids["[SEP]"]]
    h = m.enc(input_ids=torch.tensor([ids], device=m.dev)).last_hidden_state[0].float().mean(0)
    return torch.nn.functional.normalize(h, dim=0).cpu().numpy()


def main():
    m = Dettin.load(ROOT / "out" / "packs" / "ettin68-base-t4-w8L63")
    test = {w: load_cases(w, "test")[:N_TEST] for w in WORKFLOWS}
    cent, tvec = {}, {}
    for w in WORKFLOWS:
        c = np.mean([state_vec(m, x["state"]) for x in load_cases(w, "train")[:N_CENT]], 0)
        cent[w] = c / np.linalg.norm(c)
        tvec[w] = np.stack([state_vec(m, x["state"]) for x in test[w]])
        print("embedded", w, flush=True)
    probs = {}                                         # (spike, workflow) -> list over cases of {q: [p per option]}
    for sp in WORKFLOWS:
        temps = json.loads((ROOT / "results/a/spikes" / f"S-lora-{sp}-s0.json").read_text()).get("temps")
        m.load_spike(ROOT / "out/packs/adapters68" / f"S-lora-{sp}-s0.pt", temps)
        for w in WORKFLOWS:
            probs[(sp, w)] = [m.decide(c["state"], c["questions"]) for c in test[w]]
            print("spike", sp, "on", w, flush=True)
    m.unload_spike()

    out = {"tau": TAU, "by_workflow": {}}
    for w in WORKFLOWS:
        others = [s for s in WORKFLOWS if s != w]
        sims = np.stack([tvec[w] @ cent[s] for s in others], 1)          # [cases, 3]
        wts = np.exp((sims - sims.max(1, keepdims=True)) / TAU)
        wts /= wts.sum(1, keepdims=True)
        hits = {k: [] for k in ["own_spike", "uniform", "weighted", "nearest", "most_common_train"] + [f"single:{s}" for s in others]}
        train = load_cases(w, "train")
        for i, c in enumerate(test[w]):
            for q in c["questions"]:
                g = c["gold"][q]["probabilities"]
                opts = list(g)
                gold = max(g, key=g.get)
                P = np.array([[probs[(s, w)][i][q][o] for o in opts] for s in others])
                top = lambda v: opts[int(np.argmax(v))]   # noqa: E731
                hits["uniform"].append(top(P.mean(0)) == gold)
                hits["weighted"].append(top(wts[i] @ P) == gold)
                hits["nearest"].append(top(P[int(np.argmax(sims[i]))]) == gold)
                for k, s in enumerate(others):
                    hits[f"single:{s}"].append(top(P[k]) == gold)
                own = probs[(w, w)][i][q]
                hits["own_spike"].append(max(own, key=own.get) == gold)
                tops = [max(t["gold"][q]["probabilities"], key=t["gold"][q]["probabilities"].get) for t in train if q in t["gold"]]
                hits["most_common_train"].append(bool(tops) and max(set(tops), key=tops.count) == gold)
        acc = {k: round(float(np.mean(v)), 4) for k, v in hits.items()}
        acc["best_single_after_the_fact"] = max(acc[f"single:{s}"] for s in others)
        acc["mean_weight_on_nearest"] = round(float(wts.max(1).mean()), 3)
        acc["nearest_spike_share"] = {s: round(float(np.mean(np.argmax(sims, 1) == k)), 3) for k, s in enumerate(others)}
        out["by_workflow"][w] = acc
        print(w, acc, flush=True)
    keys = ["own_spike", "uniform", "weighted", "nearest", "best_single_after_the_fact", "most_common_train"]
    out["mean"] = {k: round(float(np.mean([out["by_workflow"][w][k] for w in WORKFLOWS])), 4) for k in keys}
    out["reference"] = {"full_fine_tune_on_the_other_three_68m": 0.4135, "prior_from_dataset_card": 0.470}
    (ROOT / "results" / "triangulate.json").write_text(json.dumps(out, indent=1) + "\n")
    print("MEAN", out["mean"], flush=True)
    print("done -> results/triangulate.json", flush=True)


if __name__ == "__main__":
    main()
