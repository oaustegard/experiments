"""Which sublayers, and which MLP neurons, write the loop direction?

    python3 writers.py --layer 33 --out results/writers.json

Same positions as loopdir.py (dev answers: inside terminal loops vs answers that ended).
For every layer's attention output and MLP output: mean projection on u (unit loop direction read
at --layer) at loop positions minus at normal positions. For every MLP neuron j:
act_j * (W_down[:, j] . u), the same difference of means: the neurons whose columns carry the
loop signal. The MIT edit on Qwen3.5-0.8B removed u from 64 down_proj columns in two layers.
"""
import argparse
import json
import random

import torch

from common import HERE, load, loop_onset, loop_period, prompt
from run_eval import question

FX = {x["uid"]: x for s in ("dev", "test") for x in json.loads((HERE / "data/fixture.json").read_text())[s]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--out", default="results/writers.json")
    ap.add_argument("--perp", action="store_true", help="the loop direction with the sink direction removed")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(0)
    tok, model = load()
    D = torch.load(HERE / "models/loopdir.pt")
    u = (D["dirs_perp33"] if a.perp else D["dirs"][a.layer]).float()
    L = len(model.model.layers)
    cap = {}

    def mk(name):
        def f(mod, inp, out):
            cap[name] = out[0]
            if name.startswith("mlp"):
                cap["act" + name[3:]] = inp[0][0]
        return f

    for l, lyr in enumerate(model.model.layers):
        lyr.self_attn.o_proj.register_forward_hook(mk(f"att{l}"))
        lyr.mlp.down_proj.register_forward_hook(mk(f"mlp{l}"))
    Wu = torch.stack([lyr.mlp.down_proj.weight.float().T @ u for lyr in model.model.layers])   # [L, 768]
    sums = {k: {"att": torch.zeros(L), "mlp": torch.zeros(L), "neu": torch.zeros(L, 768), "n": 0} for k in ("loop", "normal")}
    recs = [json.loads(l) for l in open(HERE / "runs/base__dev.jsonl")]
    for r in recs:
        p = tok.encode(prompt(question(FX[r["uid"]])), add_special_tokens=False)
        g = tok.encode(r["text"], add_special_tokens=False)
        if r["stopped"]:
            kind, g = "normal", g[:700]
            pos = list(range(len(p), len(p) + len(g)))
        else:
            per = loop_period(g)
            if per is None:
                continue
            s = loop_onset(g, per)
            kind, g = "loop", g[:s + 3 * per + 128]
            pos = list(range(len(p) + s + 2 * per, len(p) + len(g)))
        if not pos:
            continue
        pos = rng.sample(pos, min(len(pos), 200))
        model(torch.tensor([p + g]))
        S = sums[kind]
        for l in range(L):
            S["att"][l] += (cap[f"att{l}"][pos] @ u).sum()
            S["mlp"][l] += (cap[f"mlp{l}"][pos] @ u).sum()
            S["neu"][l] += (cap[f"act{l}"][pos] * Wu[l]).sum(0)
        S["n"] += len(pos)
    d = {k: (sums["loop"][k] / sums["loop"]["n"] - sums["normal"][k] / sums["normal"]["n"]) for k in ("att", "mlp", "neu")}
    top = torch.topk(d["neu"].flatten(), 256)
    out = {"layer_dir": a.layer, "n": {k: sums[k]["n"] for k in sums},
           "att": d["att"].tolist(), "mlp": d["mlp"].tolist(),
           "top_neurons": [[int(i) // 768, int(i) % 768, float(v)] for v, i in zip(top.values, top.indices)],
           "mlp_layer_top64_share": []}
    for l in range(L):
        v = d["neu"][l]
        out["mlp_layer_top64_share"].append(float(torch.topk(v, 64).values.sum() / v.sum()) if v.sum() > 0 else None)
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    for l in range(L):
        print(l, round(float(d["att"][l]), 2), round(float(d["mlp"][l]), 2))


if __name__ == "__main__":
    main()
