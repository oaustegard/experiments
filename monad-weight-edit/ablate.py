"""Project a direction out of everything the layers write to the residual stream.

The edit, in weights:  W_o <- (I - a u u^T) W_o  and  W_down <- (I - a u u^T) W_down  in every layer
(a = 1 removes the direction entirely; Arditi et al. 2024 weight orthogonalization). The embedding,
tied to the LM head, is left alone. As forward hooks on o_proj and down_proj outputs the same edit
is exact, so candidates are screened with hooks and only the chosen one is baked.

    python3 ablate.py screen --layer 40 [--alpha 1] [--random]   # dev loop rate / accuracy
    python3 ablate.py bake --layer 40 --alpha 1 --out models/edited  # write a bf16 HF dir
"""
import argparse
import json
import shutil

import torch

from common import HERE, MODEL_DIR, generate, load, loop_period, prompt, split_think
from run_eval import question

FX = {x["uid"]: x for x in json.loads((HERE / "data/fixture.json").read_text())["dev"]}


def direction(layer, random=False, perp=False):
    D = torch.load(HERE / "models/loopdir.pt")
    d = (D["dirs_perp33"] if perp else D["dirs"][layer]).float()
    if random:
        g = torch.Generator().manual_seed(layer)
        d = torch.randn(d.shape, generator=g)
    return d / d.norm()


def hooks(model, u, alpha):
    hs = []

    def f(mod, inp, out):
        return out - alpha * (out @ u)[..., None] * u

    for lyr in model.model.layers:
        hs.append(lyr.self_attn.o_proj.register_forward_hook(f))
        hs.append(lyr.mlp.down_proj.register_forward_hook(f))
    return hs


def bake(model, u, alpha):
    P = torch.eye(u.numel()) - alpha * torch.outer(u, u)
    with torch.no_grad():
        for lyr in model.model.layers:
            for lin in (lyr.self_attn.o_proj, lyr.mlp.down_proj):
                w = lin.weight.float()           # [256, in]: rows are residual dims
                lin.weight.copy_((P @ w).to(lin.weight.dtype))


def bake_targeted(model, u, cols, oproj, alpha=1.0):
    """Remove u only where it is written: the listed down_proj columns (layer:neuron) and the full
    o_proj of the listed layers. Everything else is untouched."""
    with torch.no_grad():
        for l, j in cols:
            w = model.model.layers[l].mlp.down_proj.weight      # [256, 768]; column j is neuron j's write
            c = w[:, j].float()
            w[:, j] = (c - alpha * (c @ u) * u).to(w.dtype)
        P = torch.eye(u.numel()) - alpha * torch.outer(u, u)
        for l in oproj:
            w = model.model.layers[l].self_attn.o_proj.weight
            w.copy_((P @ w.float()).to(w.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "bake"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--perp", action="store_true", help="loop direction minus its attention-sink component")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--out", default="models/edited")
    ap.add_argument("--cols", default="", help="layer:neuron,... (targeted edit)")
    ap.add_argument("--oproj", default="", help="layers whose o_proj loses u (targeted edit)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    u = direction(a.layer, a.random, a.perp)
    if a.cmd == "bake":
        tok, model = load(dtype=torch.bfloat16)
        cols = [tuple(map(int, c.split(":"))) for c in a.cols.split(",") if c]
        oproj = [int(x) for x in a.oproj.split(",") if x]
        if cols or oproj:
            bake_targeted(model, u, cols, oproj, a.alpha)
        else:
            bake(model, u, a.alpha)
        out = HERE / a.out
        shutil.rmtree(out, ignore_errors=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        for f in ("chat_template.json", "generation_config.json"):
            shutil.copy(MODEL_DIR / f, out / f)
        (out / "edit.json").write_text(json.dumps({"layer": a.layer, "alpha": a.alpha, "perp": a.perp, "cols": a.cols, "oproj": a.oproj, "u": u.tolist()}))
        print("baked", out)
        return
    tok, model = load()
    hooks(model, u, a.alpha)
    base = [json.loads(l) for l in open(HERE / "runs/base__dev.jsonl")]
    loops = [r for r in base if r["loop"] is not None][: a.n // 2]
    okay = [r for r in base if r["loop"] is None][: a.n - len(loops)]
    items = loops + okay
    res = generate(tok, model, [prompt(question(FX[r["uid"]])) for r in items], max_new=a.max_new, batch=24)
    rows = []
    for r, g in zip(items, res):
        text = tok.decode(g["ids"])
        rows.append({"uid": r["uid"], "was_loop": r["loop"] is not None, "loop": (not g["stopped"]) and loop_period(g["ids"]) is not None,
                     "stopped": g["stopped"], "closed": split_think(text)[2], "n": len(g["ids"]), "text": text})
    tag = f"L{a.layer}_a{a.alpha}" + ("_rand" if a.random else "")
    (HERE / "runs").mkdir(exist_ok=True)
    with open(HERE / f"runs/screen_{tag}.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    wl = [x for x in rows if x["was_loop"]]
    ok = [x for x in rows if not x["was_loop"]]
    print(json.dumps({"tag": tag, "loops_fixed": sum(not x["loop"] for x in wl), "of": len(wl),
                      "new_loops_in_ok": sum(x["loop"] for x in ok), "of_ok": len(ok),
                      "closed_wl": sum(x["closed"] for x in wl), "stopped_wl": sum(x["stopped"] for x in wl)}))


if __name__ == "__main__":
    main()
