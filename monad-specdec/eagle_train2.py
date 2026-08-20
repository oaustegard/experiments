"""Train the EAGLE head across however many harvest shards have landed.

Usage: eagle_train2.py <n_shards> <epochs> [tag]

Held-out data is the tail of the last shard, so every data-size point is scored
against the same distribution it trained on.
"""
import glob, json, sys, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from eagle_train import EagleHead

torch.set_num_threads(4)
BATCH_SEQ, LOSS_POS, LR = 4, 128, 3e-4
N_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TAG = sys.argv[3] if len(sys.argv) > 3 else f"s{N_SHARDS}e{EPOCHS}"
VAL_SEQ = 48

paths = sorted(glob.glob("/tmp/specdec/shards/*.npz"))[:N_SHARDS]
assert paths, "no shards"
# Preallocate and fill in place. Building a list then concatenating peaks at
# twice the corpus size, which is what silently killed the first attempt.
sizes = []
for p in paths:
    with np.load(p) as d:
        sizes.append(d["tok"].shape[0])
total = sum(sizes)
dim = 576
h = np.empty((total, dim), dtype=np.float16)
tk = np.empty(total, dtype=np.int32)
lb = np.empty(total, dtype=np.int32)
off = 0
for p, n in zip(paths, sizes):
    with np.load(p) as d:
        h[off:off + n] = d["h"]; tk[off:off + n] = d["tok"]; lb[off:off + n] = d["lab"]
    off += n
    print(f"  loaded {p.split('/')[-1]} ({off}/{total})", flush=True)
SEQ = 512; SL = SEQ - 1
n_seq = h.shape[0] // SL
h = h[:n_seq * SL].reshape(n_seq, SL, -1)
tk = tk[:n_seq * SL].reshape(n_seq, SL)
lb = lb[:n_seq * SL].reshape(n_seq, SL)
n_train = n_seq - VAL_SEQ
print(f"[{TAG}] {len(paths)} shards, {n_seq} seqs, train {n_train} "
      f"({n_train*SL} tokens), val {VAL_SEQ}", flush=True)

cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                              dtype=torch.float32).eval()
embed, lm_head = target.model.embed_tokens, target.lm_head
for p_ in target.parameters():
    p_.requires_grad_(False)


def evaluate(model, n=VAL_SEQ):
    model.eval(); hit = tot = 0
    with torch.no_grad():
        for i in range(n_train, n_train + n):
            pred = model(torch.from_numpy(h[i]).float().unsqueeze(0),
                         embed(torch.from_numpy(tk[i].astype(np.int64)).unsqueeze(0)))
            for s in range(0, SL, 128):
                am = lm_head(pred[:, s:s + 128]).argmax(-1)[0]
                gold = torch.from_numpy(lb[i][s:s + 128].astype(np.int64))
                hit += int((am == gold).sum()); tot += len(gold)
    model.train(); return hit / tot


torch.manual_seed(0)
head = EagleHead(cfg)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
order0 = np.arange(n_train)
curve, steps, t0 = [], 0, time.perf_counter()
best = 0.0
for ep in range(EPOCHS):
    order = order0.copy(); np.random.shuffle(order)
    for b in range(0, len(order) - BATCH_SEQ + 1, BATCH_SEQ):
        idx = order[b:b + BATCH_SEQ]
        with torch.no_grad():
            emb = embed(torch.from_numpy(tk[idx].astype(np.int64)))
        pred = head(torch.from_numpy(h[idx]).float(), emb)
        sel = torch.randint(0, SL, (LOSS_POS,))
        lg = lm_head(pred[:, sel])
        gold = torch.from_numpy(lb[idx][:, sel.numpy()].astype(np.int64))
        loss = nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                           gold.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
        opt.step(); steps += 1
        if steps % 50 == 0:
            open("/tmp/specdec/heartbeat", "w").write(
                f"{time.time()} {TAG} step{steps}\n")
    a = evaluate(head)
    curve.append({"epoch": ep + 1, "acceptance": round(a, 4),
                  "loss": round(loss.item(), 3),
                  "min": round((time.perf_counter() - t0) / 60, 1)})
    print(f"[{TAG}] ep{ep+1} alpha={a:.4f} loss={loss.item():.3f} "
          f"{(time.perf_counter()-t0)/60:.1f}min", flush=True)
    open("/tmp/specdec/heartbeat", "w").write(
        f"{time.time()} {TAG} ep{ep+1} alpha={a:.4f}\n")
    if a > best:
        best = a
        torch.save(head.state_dict(), f"/tmp/specdec/eagle_head_{TAG}.pt")

out = {"tag": TAG, "shards": len(paths), "train_tokens": int(n_train * SL),
       "epochs": EPOCHS, "steps": steps, "best_acceptance": round(best, 4),
       "curve": curve, "minutes": round((time.perf_counter() - t0) / 60, 1)}
print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
json.dump(out, open(f"eagle_train_{TAG}.json", "w"), indent=2)
