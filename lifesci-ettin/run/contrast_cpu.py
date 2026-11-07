"""Step 0 of the contrastive round: the kill test, on CPU in PyTorch (no WGSL).

    python3 -I run/contrast_cpu.py NAME START ARM [--steps 1000] [--lr 5e-5]
      START: "e17c" (the SentenceTransformer dir) or a flat .f32 under .work/shards/
      ARM:   stock | ls32 (which shards and table)

Loss: MultipleNegativesRankingLoss, the recipe that made e17c (ettin-native/run/contrast.py): in-batch
negatives, scale 20 (temperature 0.05), query -> passage only, AdamW wd 0.01, linear warmup then decay.
Pairs and their order come from run/pairs.py (64 pairs a step), so both arms see the same batches.
Writes .work/ckpt/NAME.f32 and results/train/NAME.summary.json (loss per step).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import Embedder, from_flat, load_st, to_flat  # noqa: E402

W = HERE / ".work"
S = W / "shards"
torch.set_num_threads(4)
PAD = {"stock": 50283, "ls32": None}


def batch(ids, offs, lens, rows, pad):
    n = int(lens[rows].max())
    out = torch.full((len(rows), n), pad, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, r in enumerate(rows):
        l = int(lens[r])
        out[i, :l] = torch.from_numpy(ids[offs[r]:offs[r] + l].astype(np.int64))
        att[i, :l] = 1
    return out, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("start"); ap.add_argument("arm")
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warm", type=int, default=100); ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--pad", type=int)
    ap.add_argument("--shards", help="shard prefix under .work/shards (default ARM)")
    ap.add_argument("--pairs", default="pairs.npz")
    a = ap.parse_args()
    sh = a.shards or a.arm
    lens = np.fromfile(S / f"{sh}.lens.u16", np.uint16).astype(np.int64)
    ids = np.fromfile(S / f"{sh}.ids.u16", np.uint16)
    offs = np.concatenate([[0], np.cumsum(lens)])
    pad = a.pad if a.pad is not None else PAD[a.arm]
    order = np.load(S / a.pairs)["order"]
    P = 64
    if a.start == "e17c":
        p = load_st(W / "e17c/epoch1")
    else:
        flat = np.fromfile(S / a.start, np.float32)
        p = from_flat(flat, (len(flat) - 4099840) // 256)
    m = Embedder(p).train()
    ps = [t for _, t in m.ordered()]
    opt = torch.optim.AdamW(ps, lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warm) * max(0.0, (a.steps - s) / max(1, a.steps - a.warm)))
    losses, acc, t0 = [], [], time.time()
    for s in range(a.steps):
        pr = order[s * P:(s + 1) * P]
        qi, qa = batch(ids, offs, lens, pr[:, 0], pad)
        di, da = batch(ids, offs, lens, pr[:, 1], pad)
        q = F.normalize(m(qi, qa), dim=-1)
        d = F.normalize(m(di, da), dim=-1)
        sim = a.scale * q @ d.T
        loss = F.cross_entropy(sim, torch.arange(P))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        sch.step()
        losses.append(float(loss))
        acc.append(float((sim.argmax(1) == torch.arange(P)).float().mean()))
        if (s + 1) % 25 == 0:
            el = time.time() - t0
            print(f"step {s + 1}/{a.steps} loss {np.mean(losses[-25:]):.4f} acc {np.mean(acc[-25:]):.3f} {el / (s + 1):.2f}s/step", flush=True)
    (W / "ckpt").mkdir(parents=True, exist_ok=True)
    to_flat(m).tofile(W / "ckpt" / f"{a.name}.f32")
    out = {"name": a.name, "start": a.start, "arm": a.arm, "shards": sh, "pairs": a.pairs, "steps": a.steps, "lr": a.lr, "scale": a.scale, "P": P,
           "s_per_step": (time.time() - t0) / a.steps, "loss_first100": float(np.mean(losses[:100])),
           "loss_last100": float(np.mean(losses[-100:])), "acc_last100": float(np.mean(acc[-100:])), "losses": [round(x, 4) for x in losses]}
    (HERE / "results/train").mkdir(parents=True, exist_ok=True)
    (HERE / "results/train" / f"{a.name}.summary.json").write_text(json.dumps(out))
    print({k: v for k, v in out.items() if k != "losses"})


if __name__ == "__main__":
    main()
