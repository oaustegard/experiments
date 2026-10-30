"""Fine-tune a conv TRM on several grid sizes at once, with no position table, so one model plans on any
grid (a 32×32 window or a whole level).  python train_multi.py DATA OUT --init CKPT [--key value ...]

Same deep supervision with a carry as train.py, one carry per grid size; each optimizer step takes the
next size in turn. Sizes come as n:batch, data for n=32 from DATA/train*, for others from DATA/b{n}*.
"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trm import TRM

p = argparse.ArgumentParser()
p.add_argument("data"); p.add_argument("out"); p.add_argument("--init", required=True)
p.add_argument("--sizes", default="32:256,48:112,64:64"); p.add_argument("--nsup", type=int, default=12)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--steps", type=int, default=30000); p.add_argument("--warmup", type=int, default=300)
p.add_argument("--ema", type=float, default=0.999); p.add_argument("--eval_every", type=int, default=1500)
p.add_argument("--explore", type=float, default=0.1); p.add_argument("--hours", type=float, default=4)
p.add_argument("--compile", type=int, default=1)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
root = Path(a.data)
sizes = [(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.sizes.split(",")]

def load(dirs):
    return (torch.from_numpy(np.concatenate([np.load(d / "x.npy") for d in dirs])).to(dev).long(),
            torch.from_numpy(np.concatenate([np.load(d / "y.npy") for d in dirs])).to(dev).long())
data = {n: load(sorted(root.glob("train*")) if n == 32 else sorted(root.glob(f"b{n}[a-z]"))) for n, _ in sizes}
for n in data: print("size", n, "samples", len(data[n][0]), flush=True)
evals = {"val32": (32, *load([root / "val"])), "test64_E1M9": (64, *load([root / "b64test"]))}

ck = torch.load(a.init, map_location=dev); ia = ck["args"]
model = TRM(L=1024, d=ia["d"], heads=ia["heads"], layers=ia["layers"], n=ia["n"], T=ia["T"], mixer="conv").to(dev)
model.load_state_dict(ck["ema"])
with torch.no_grad(): model.pos.zero_()
model.pos.requires_grad_(False)
ema = copy.deepcopy(model).eval()
for q in ema.parameters(): q.requires_grad_(False)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
net_eager = model.net
if a.compile: model.net = torch.compile(model.net, dynamic=False)

def embed(m, x):  # no positions, whatever the size
    return m.emb(x)

def dihedral(x, n, k):
    g = x.view(-1, n, n)
    if k & 4: g = g.transpose(1, 2)
    return torch.rot90(g, k & 3, (1, 2)).reshape(-1, n * n)

def draw(n, m):
    X, Y = data[n]; idx = torch.randint(0, len(X), (m,), device=dev); k = int(torch.randint(0, 8, ()))
    return dihedral(X[idx], n, k), dihedral(Y[idx], n, k)

def set_grid(m, n):
    for b in m.blocks: b.hw = (n, n)

state = {}
for n, B in sizes:
    x, yt = draw(n, B); y, z = model.init_state(B, n * n, dev, torch.float32)
    state[n] = {"x": x, "yt": yt, "y": y, "z": z, "depth": torch.zeros(B, dtype=torch.long, device=dev), "B": B}

@torch.no_grad()
def evaluate(m):
    res = {}
    for name, (n, X, Y) in evals.items():
        set_grid(m, n); X, Y = X[:600], Y[:600]; ex = []
        for b in range(0, len(X), 64):
            x, yt = X[b:b + 64], Y[b:b + 64]
            with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
                h = embed(m, x); y, z = m.init_state(len(x), n * n, dev, h.dtype); e = []
                for s in range(a.nsup):
                    y, z, lg, q = m.step(h, y, z); e.append((lg.argmax(-1) == yt).all(-1))
            ex.append(torch.stack(e, 1))
        E = torch.cat(ex).float().mean(0)
        res[name] = {f"@{s + 1}": round(E[s].item(), 3) for s in sorted({0, 1, 3, 5, 7, a.nsup - 1}) if s < a.nsup}
    return res

def lr_at(s):
    if s < a.warmup: return a.lr * s / a.warmup
    return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))))

log = open(out / "log.jsonl", "a"); t0 = time.time(); acc = {}
for step in range(a.steps):
    n, B = sizes[step % len(sizes)]; S = state[n]
    set_grid(model, n)
    for g in opt.param_groups: g["lr"] = lr_at(step)
    with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
        h = embed(model, S["x"])
        y, z, lg, q = model.step(h, S["y"], S["z"])
        loss = F.cross_entropy(lg.float().reshape(-1, 2), S["yt"].reshape(-1))
        exact = (lg.argmax(-1) == S["yt"]).all(-1)
        qloss = F.binary_cross_entropy_with_logits(q.float(), exact.float())
    opt.zero_grad(set_to_none=True); (loss + 0.5 * qloss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - a.ema)
        S["y"], S["z"] = y, z; S["depth"] += 1
        stop = (S["depth"] >= a.nsup) | ((q > 0) & (torch.rand(B, device=dev) > a.explore))
        if stop.any():
            k = int(stop.sum()); nx, ny = draw(n, k)
            S["x"], S["yt"] = S["x"].clone(), S["yt"].clone(); S["x"][stop], S["yt"][stop] = nx, ny
            y0, z0 = model.init_state(k, n * n, dev, y.dtype)
            S["y"], S["z"] = S["y"].clone(), S["z"].clone(); S["y"][stop], S["z"][stop] = y0, z0; S["depth"][stop] = 0
    a_ = acc.setdefault(n, [0.0, 0.0, 0]); a_[0] += loss.item(); a_[1] += exact.float().mean().item(); a_[2] += 1
    if (step + 1) % 150 == 0:
        r = {"step": step + 1, "lr": round(lr_at(step), 6), "min": round((time.time() - t0) / 60, 1)}
        for nn_, (l, e, c) in acc.items(): r[f"loss{nn_}"] = round(l / c, 4); r[f"exact{nn_}"] = round(e / c, 3)
        acc = {}; print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    last = step + 1 == a.steps or (time.time() - t0) / 3600 > a.hours
    if (step + 1) % a.eval_every == 0 or last:
        r = {"step": step + 1, "eval": evaluate(ema), "min": round((time.time() - t0) / 60, 1)}
        print("EVAL " + json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        args = dict(ia); args.update(mixer="conv", nsup=a.nsup, multi=a.sizes, nopos=True)
        torch.save({"ema": ema.state_dict(), "args": args, "step": step + 1}, out / "ema.pt.tmp"); os.replace(out / "ema.pt.tmp", out / "ema.pt")
    if last: break
print("DONE", flush=True)
