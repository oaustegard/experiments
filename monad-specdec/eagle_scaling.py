"""Is the drafter limited by data or by compute?

Two arms on the same 225k-token harvest.

  Arm A, more passes over fixed data: train on everything, measure acceptance
  after each epoch. If repetition alone keeps lifting alpha, compute is the
  constraint.

  Arm B, more data at fixed passes: train on 25%, 50% and 100% of the corpus
  for the same number of epochs. The slope of alpha against corpus size says
  what another order of magnitude of harvest would buy.

The distinction decides whether a GPU is needed and what for.
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from eagle_train import EagleHead

torch.set_num_threads(4)

DATA = "/tmp/specdec/eagle_data.npz"
BATCH_SEQ, LOSS_POS, LR = 4, 128, 3e-4

d = np.load(DATA)
h, tk, lb, seq = d["h"], d["tok"], d["lab"], int(d["seq"])
L = seq - 1
n_seq = h.shape[0] // L
h = h[:n_seq * L].reshape(n_seq, L, -1)
tk = tk[:n_seq * L].reshape(n_seq, L)
lb = lb[:n_seq * L].reshape(n_seq, L)
n_val = 48
n_train_all = n_seq - n_val

cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                              dtype=torch.float32).eval()
embed, lm_head = target.model.embed_tokens, target.lm_head
for p in target.parameters():
    p.requires_grad_(False)


def evaluate(model, n=16):
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for i in range(n_seq - n_val, n_seq - n_val + n):
            pred = model(torch.from_numpy(h[i]).float().unsqueeze(0),
                         embed(torch.from_numpy(tk[i].astype(np.int64)).unsqueeze(0)))
            for s in range(0, L, 128):
                am = lm_head(pred[:, s:s + 128]).argmax(-1)[0]
                gold = torch.from_numpy(lb[i][s:s + 128].astype(np.int64))
                hit += int((am == gold).sum()); tot += len(gold)
    model.train()
    return hit / tot


def train(n_train_seq, epochs, tag, eval_every_epoch=False):
    torch.manual_seed(0)
    head = EagleHead(cfg)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    order0 = np.arange(n_train_seq)
    curve, steps, t0 = [], 0, time.perf_counter()
    for ep in range(epochs):
        order = order0.copy(); np.random.shuffle(order)
        for b in range(0, len(order) - BATCH_SEQ + 1, BATCH_SEQ):
            idx = order[b:b + BATCH_SEQ]
            with torch.no_grad():
                emb = embed(torch.from_numpy(tk[idx].astype(np.int64)))
            pred = head(torch.from_numpy(h[idx]).float(), emb)
            sel = torch.randint(0, L, (LOSS_POS,))
            lg = lm_head(pred[:, sel])
            gold = torch.from_numpy(lb[idx][:, sel.numpy()].astype(np.int64))
            loss = nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                               gold.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
            opt.step(); steps += 1
        if eval_every_epoch or ep == epochs - 1:
            a = evaluate(head)
            curve.append({"epoch": ep + 1, "steps": steps,
                          "tokens_seen": int((ep + 1) * n_train_seq * L),
                          "acceptance": round(a, 4),
                          "loss": round(loss.item(), 3)})
            print(f"  [{tag}] ep{ep+1} alpha={a:.4f} loss={loss.item():.3f} "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    return curve, round(time.perf_counter() - t0, 1)


def speedup(a, g, c):
    return (1 - a ** (g + 1)) / ((1 - a) * (1 + g * c))


C = 0.049  # measured fp32 cost ratio for the EAGLE draft step
out = {"c_used": C, "corpus_tokens": int(n_train_all * L)}

print("ARM A - more passes over all 225k tokens", flush=True)
out["arm_a_epochs"], out["arm_a_sec"] = train(n_train_all, 16, "A", True)

print("ARM B - more data, 4 epochs each", flush=True)
arm_b = []
for frac in (0.25, 0.5, 1.0):
    n = max(BATCH_SEQ, int(n_train_all * frac))
    curve, sec = train(n, 4, f"B{frac}")
    arm_b.append({"fraction": frac, "train_tokens": int(n * L),
                  "acceptance": curve[-1]["acceptance"], "sec": sec})
out["arm_b_data"] = arm_b

best = max(out["arm_a_epochs"], key=lambda r: r["acceptance"])
out["best_acceptance"] = best
out["projected_speedup_at_best"] = {
    f"gamma={g}": round(speedup(best["acceptance"], g, C), 3) for g in range(1, 7)}
out["monad_reference"] = {"acceptance": 0.546, "c": 0.476,
                          "measured_speedup": 0.90}
print(json.dumps({k: v for k, v in out.items() if k != "arm_a_epochs"}, indent=2))
json.dump(out, open("eagle_scaling.json", "w"), indent=2)
