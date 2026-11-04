"""Repeat monad-weight-edit's loop diagnosis on the fine-tuned model: where is its loop signal now?

    python3 rediag.py --model models/skygest_e8 --run runs/skygest_e8__bridge.jsonl

One teacher-forced pass per dev answer (the model's own greedy outputs on the Mac, `set == "dev"`), reading
the residual stream after every layer and what every attention block (o_proj output) and MLP (down_proj
output) writes. Positions, as in monad-weight-edit/loopdir.py: inside a terminal loop, two periods past
its onset ("loop"), and in answers that ended on their own ("normal").

Per layer: the difference-of-means direction d_l = mean(loop) - mean(normal), how well the projection on it
separates the two (AUROC), and its cosine with token 0's residual (the attention sink, which dominated the
direction in the original Monad). At the best layer the sink component is projected out (u_perp), and every
sublayer's write is scored by its mean projection on u_perp, loop minus normal: the writers an edit would
target. Also reports the cosine of u_perp with A8p's edited direction (models/A8p/edit.json).

Writes results/rediag.json and models/loopdir_ft.pt (all d_l, u_perp, sink direction).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("monad-weight-edit")))
from common import loop_onset, loop_period  # noqa: E402


def prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def auroc(pos, neg):
    s = torch.cat([pos, neg])
    r = torch.empty_like(s)
    r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models/skygest_e8"))
    ap.add_argument("--run", default=str(HERE / "runs/skygest_e8__bridge.jsonl"))
    ap.add_argument("--per", type=int, default=100, help="positions sampled per answer")
    ap.add_argument("--out", default="results/rediag.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    L = len(model.model.layers)
    ev = {r["id"]: r for r in map(json.loads, open(HERE / "data/eval_prompts.jsonl"))}
    rows = [r for r in map(json.loads, open(a.run)) if r.get("set") == "dev"]
    cap = {}
    for l, lyr in enumerate(model.model.layers):
        lyr.self_attn.o_proj.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(f"att{l}", o[0]))
        lyr.mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(f"mlp{l}", o[0]))
    H = {"loop": [], "normal": []}
    # attention and MLP writes summed over the sampled positions: the writer gap along any direction is linear in the mean
    W = {"loop": torch.zeros(L, 2, 256, dtype=torch.float64), "normal": torch.zeros(L, 2, 256, dtype=torch.float64)}
    Wn = {"loop": 0, "normal": 0}
    sink = []
    meta = {"loop": 0, "normal": 0}
    for r in rows:
        p = tok.encode(prompt(ev[r["id"]]["q"]), add_special_tokens=False)
        text = r["trace"] if not r["closed"] else r["trace"] + "</think>\n\n" + r["answer"]
        g = tok.encode(text, add_special_tokens=False)
        if r["stopped"]:
            kind, g = "normal", g[:700]
            pos = list(range(len(p), len(p) + len(g)))
        else:
            per = loop_period(g)
            if not r["loop"] or per is None:
                continue
            s = loop_onset(g, per)
            kind, g = "loop", g[:s + 3 * per + 128]
            pos = list(range(len(p) + s + 2 * per, len(p) + len(g)))
        if not pos:
            continue
        out = model(torch.tensor([p + g]), output_hidden_states=True)
        hs = torch.stack([h[0] for h in out.hidden_states[1:]])         # [L, T, 256], after each layer
        sel = rng.sample(pos, min(len(pos), a.per))
        H[kind].append(hs[:, sel].transpose(0, 1))                       # [n, L, 256]
        W[kind] += torch.stack([torch.stack([cap[f"att{l}"][sel].sum(0), cap[f"mlp{l}"][sel].sum(0)]) for l in range(L)]).double()
        Wn[kind] += len(sel)
        sink.append(hs[:, 0])                                            # token 0 after every layer
        meta[kind] += 1
    Hc = {k: torch.cat(v) for k, v in H.items()}
    Wm = {k: (W[k] / Wn[k]).float() for k in W}
    sinkm = torch.stack(sink).mean(0)                                    # [L, 256]
    layers, dirs = [], []
    for l in range(L):
        d = Hc["loop"][:, l].mean(0) - Hc["normal"][:, l].mean(0)
        u = d / d.norm()
        dirs.append(u)
        sd = sinkm[l] / sinkm[l].norm()
        layers.append({"layer": l, "auroc": auroc(Hc["loop"][:, l] @ u, Hc["normal"][:, l] @ u),
                       "gap": float(d.norm()), "resid_norm": float(Hc["normal"][:, l].norm(dim=-1).mean()),
                       "cos_sink": float(u @ sd)})
    best = max(range(8, L), key=lambda l: layers[l]["auroc"])
    u = dirs[best]
    sd = sinkm[best] / sinkm[best].norm()
    up = u - (u @ sd) * sd
    up = up / up.norm()
    perp_auroc = [auroc(Hc["loop"][:, l] @ up, Hc["normal"][:, l] @ up) for l in range(L)]
    a8p = torch.tensor(json.loads((HERE / "models/A8p/edit.json").read_text())["u"], dtype=torch.float32)
    writers = []
    for l in range(L):
        for j, kind in enumerate(("att", "mlp")):
            gap = float((Wm["loop"][l, j] - Wm["normal"][l, j]) @ up)
            writers.append({"layer": l, "kind": kind, "gap": gap})
    writers.sort(key=lambda x: -abs(x["gap"]))
    res = {"model": a.model, "run": a.run, "answers": meta, "positions": {k: int(v.shape[0]) for k, v in Hc.items()},
           "best_layer": best, "best_auroc": layers[best]["auroc"], "best_cos_sink": layers[best]["cos_sink"],
           "perp_auroc_at_best": perp_auroc[best], "cos_perp_a8p_u": float(abs(up @ a8p / a8p.norm())),
           "cos_raw_a8p_u": float(abs(u @ a8p / a8p.norm())),
           "total_gap_on_perp": float((Hc["loop"][:, best] @ up).mean() - (Hc["normal"][:, best] @ up).mean()),
           "layers": layers, "perp_auroc": perp_auroc, "writers_top": writers[:20]}
    (HERE / a.out).write_text(json.dumps(res, indent=1))
    torch.save({"dirs": torch.stack(dirs), "u_perp": up, "sink": sd, "best_layer": best}, HERE / "models/loopdir_ft.pt")
    print(json.dumps({k: v for k, v in res.items() if k not in ("layers", "perp_auroc", "writers_top")}, indent=1))
    for l in range(0, L, 8):
        print(layers[l])
    for w in writers[:12]:
        print(w)


if __name__ == "__main__":
    main()
