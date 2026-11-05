"""System v0.1.0 design check: can the kNN key come from the spike's own pass instead of a second model?

    python3 -I run/key_check.py

For each workflow, the round 1 LoRA spike (seed 0) runs each case's first question; the key is the mean
of its final hidden states over the state tokens. The kNN growth curve with these keys is compared with
round 1's e17c-embedder keys (results/knn.json). Writes results/key_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import add_lora, load_adapter, stock_path  # noqa: E402
from shared import KNN_SEEDS, ROOT, SIZES, WORKFLOWS, examples, load_cases, revealed, score_examples  # noqa: E402
from train_enc import Decider, Enc  # noqa: E402

K, TAU, EPS = 8, 0.05, 0.01


@torch.no_grad()
def keys_for(w, tok):
    model = add_lora(Decider(stock_path())).eval()
    load_adapter(model, torch.load(ROOT / "out" / "adapters" / f"S-lora-{w}-s0.pt"))
    out = {}
    for split in ("train", "test"):
        for c in load_cases(w, split):
            e = examples([c])[0]
            ids, pos = tok.build(e)
            start = ids.index(tok.sep, pos[-1]) + 1          # state tokens follow the options' [SEP]
            h = model.enc(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
            v = h[start: len(ids) - 1].mean(0)
            out[c["id"]] = torch.nn.functional.normalize(v, dim=0).numpy()
    return out


def main():
    torch.set_num_threads(4)
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    e17c = json.loads((ROOT / "results" / "knn.json").read_text())["runs"]
    res = {}
    for w in WORKFLOWS:
        E = keys_for(w, tok)
        te = examples(load_cases(w, "test"))
        for n in SIZES[1:]:
            accs = []
            for seed in KNN_SEEDS:
                by_q = {}
                for e in examples(revealed(w, seed, n)):
                    by_q.setdefault(e["qname"], []).append((E[e["case"]], np.array(e["gold"])))
                probs = []
                for e in te:
                    keys = np.stack([k for k, _ in by_q[e["qname"]]])
                    golds = np.stack([g for _, g in by_q[e["qname"]]])
                    s = keys @ E[e["case"]]
                    top = np.argsort(-s)[: min(K, len(s))]
                    wt = np.exp((s[top] - s[top].max()) / TAU)
                    p = (wt[:, None] * golds[top]).sum(0) / wt.sum()
                    probs.append(((1 - EPS) * p + EPS / len(p)).tolist())
                accs.append(score_examples(te, probs)[0]["accuracy"])
            res[f"{w}|n{n}"] = {"spike_key": float(np.mean(accs)),
                                "e17c_key": float(np.mean([e17c[f"{w}|s{s}|n{n}"]["accuracy"] for s in KNN_SEEDS]))}
            print(w, n, res[f"{w}|n{n}"], flush=True)
    mean = {f"n{n}": {k: float(np.mean([res[f"{w}|n{n}"][k] for w in WORKFLOWS])) for k in ("spike_key", "e17c_key")}
            for n in SIZES[1:]}
    (ROOT / "results" / "key_check.json").write_text(json.dumps({"mean": mean, "runs": res}, indent=1))
    print(json.dumps(mean, indent=1))


if __name__ == "__main__":
    main()
