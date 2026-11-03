"""Is there a "loop direction" in Monad's residual stream, as there was in Qwen3.5-0.8B (MIT post)?

    python3 loopdir.py --run runs/base__dev.jsonl --out results/loopdir.json

Teacher-forces prompt + generation for every dev answer and reads the residual stream after
every layer at three kinds of position:
  loop    inside a terminal loop, two periods past its onset
  pre     the 64 tokens before a loop's onset
  normal  answers that ended on their own
Per layer: the difference-of-means direction loop - normal, how well the projection on it
separates the two (AUROC), and the same for pre vs normal (could the direction predict a loop
before it starts?). Saves the unit directions to models/loopdir.pt for the intervention test.
"""
import argparse
import json
import random

import torch

from common import HERE, load, loop_onset, loop_period, prompt
from run_eval import question

FX = {x["uid"]: x for s in ("dev", "test") for x in json.loads((HERE / "data/fixture.json").read_text())[s]}


def auroc(pos, neg):
    """P(score_pos > score_neg), ties half."""
    s = torch.cat([pos, neg])
    r = torch.empty_like(s)
    r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/base__dev.jsonl")
    ap.add_argument("--out", default="results/loopdir.json")
    ap.add_argument("--per", type=int, default=3000)
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = random.Random(0)
    tok, model = load()
    recs = [json.loads(l) for l in open(HERE / a.run)]
    acts = {"loop": [], "pre": [], "normal": []}
    meta = []
    for r in recs:
        p = tok.encode(prompt(question(FX[r["uid"]])), add_special_tokens=False)
        g = tok.encode(r["text"], add_special_tokens=False)
        if r["stopped"]:
            kind = "normal"
            g = g[:700]
            pos = list(range(len(p), len(p) + len(g)))
        else:
            per = loop_period(g)
            if per is None:
                continue
            s = loop_onset(g, per)
            kind = "loop"
            g = g[:s + 3 * per + 128]     # deep enough into the loop; keeps the forward pass short
            pos = list(range(len(p) + s + 2 * per, len(p) + len(g)))
            pre = list(range(len(p) + max(0, s - 64), len(p) + s))
            meta.append({"uid": r["uid"], "period": per, "onset": s, "n": len(g)})
        hs = model(torch.tensor([p + g]), output_hidden_states=True).hidden_states   # 65 x [1, T, 256]
        H = torch.stack([h[0] for h in hs])                                          # [65, T, 256]
        take = lambda ps: H[:, rng.sample(ps, min(len(ps), 200))] if ps else None   # noqa: E731
        x = take(pos)
        if x is not None:
            acts[kind].append(x)
        if kind == "loop" and pre:
            acts["pre"].append(take(pre))
    A = {k: torch.cat(v, 1) for k, v in acts.items() if v}
    for k in A:
        if A[k].shape[1] > a.per:
            A[k] = A[k][:, torch.randperm(A[k].shape[1])[:a.per]]
    out = {"counts": {k: int(v.shape[1]) for k, v in A.items()}, "loops": meta, "layers": []}
    dirs = []
    for l in range(A["loop"].shape[0]):
        lo, no = A["loop"][l], A["normal"][l]
        d = lo.mean(0) - no.mean(0)
        u = d / d.norm()
        dirs.append(u)
        row = {"layer": l, "norm_gap": float(d.norm()), "resid_norm": float(no.norm(dim=-1).mean()),
               "auroc_loop": auroc(lo @ u, no @ u)}
        if "pre" in A:
            row["auroc_pre"] = auroc(A["pre"][l] @ u, no @ u)
        out["layers"].append(row)
    torch.save({"dirs": torch.stack(dirs), "counts": out["counts"]}, HERE / "models/loopdir.pt")
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    for row in out["layers"][::4]:
        print(row)
    print(out["counts"])


if __name__ == "__main__":
    main()
