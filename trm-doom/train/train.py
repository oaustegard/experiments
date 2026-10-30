"""Train TRM on the grid planning data.  python train.py DATA OUT [--key value ...]

Deep supervision with a carry, as in the paper's code: every optimizer step runs one supervision step on
a batch whose samples are at different depths; a sample leaves when the halting head says so (or at
nsup steps) and a fresh one takes its slot. Checkpoints and a JSON log go to OUT every eval.
"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trm import TRM, use_flex

p = argparse.ArgumentParser()
p.add_argument("data"); p.add_argument("out")
p.add_argument("--d", type=int, default=128); p.add_argument("--heads", type=int, default=4)
p.add_argument("--layers", type=int, default=2); p.add_argument("--n", type=int, default=4)
p.add_argument("--T", type=int, default=3); p.add_argument("--nsup", type=int, default=8)
p.add_argument("--batch", type=int, default=256); p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=0.1); p.add_argument("--steps", type=int, default=60000)
p.add_argument("--warmup", type=int, default=1000); p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--eval_every", type=int, default=2000); p.add_argument("--eval_n", type=int, default=2000)
p.add_argument("--explore", type=float, default=0.1); p.add_argument("--hours", type=float, default=99)
p.add_argument("--resume", default=""); p.add_argument("--flex", type=int, default=0)
p.add_argument("--compile", type=int, default=0); p.add_argument("--mixer", default="attn")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
torch.backends.cuda.matmul.allow_tf32 = True
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
N = 32

def load(dirs):
    xs, ys, ms = [], [], []
    for d in dirs:
        xs.append(np.load(d / "x.npy")); ys.append(np.load(d / "y.npy")); ms.append(np.load(d / "meta.npy"))
    return (torch.from_numpy(np.concatenate(xs)).long(), torch.from_numpy(np.concatenate(ys)).long(),
            torch.from_numpy(np.concatenate(ms)).long())

root = Path(a.data)
Xtr, Ytr, _ = load(sorted(root.glob("train*")))
Xva, Yva, Mva = load([root / "val"]); Xte, Yte, Mte = load([root / "test"])
Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
print(f"train {len(Xtr)} val {len(Xva)} test {len(Xte)} on {dev}", flush=True)

def dihedral(x, k):
    g = x.view(-1, N, N)
    if k & 4: g = g.transpose(1, 2)
    return torch.rot90(g, k & 3, (1, 2)).reshape(-1, N * N)

def draw(m):
    idx = torch.randint(0, len(Xtr), (m,), device=dev)
    x, y = Xtr[idx], Ytr[idx]
    k = int(torch.randint(0, 8, ()))
    return dihedral(x, k), dihedral(y, k)

if a.flex: use_flex()
model = TRM(L=N * N, d=a.d, heads=a.heads, layers=a.layers, n=a.n, T=a.T, mixer=a.mixer).to(dev)
ema = copy.deepcopy(model).eval()
for q in ema.parameters(): q.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
step0 = 0
if a.resume and Path(a.resume).exists():
    ck = torch.load(a.resume, map_location=dev)
    model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    print("resumed at", step0, flush=True)
print("params", sum(q.numel() for q in model.parameters()), flush=True)
if a.compile: model.net = torch.compile(model.net, dynamic=False)

@torch.no_grad()
def evaluate(m, X, Y, M, nmax):
    m.eval()
    X, Y, M = X[:nmax].to(dev), Y[:nmax].to(dev), M[:nmax].to(dev)
    per = []
    halted_exact = torch.zeros(len(X), dtype=torch.bool, device=dev); halted = torch.zeros_like(halted_exact)
    exact_by = []
    for b in range(0, len(X), 256):
        x, yt = X[b:b + 256], Y[b:b + 256]
        with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
            h = m.embed(x); y, z = m.init_state(len(x), N * N, dev, h.dtype)
            ex = []
            for s in range(a.nsup):
                y, z, lg, q = m.step(h, y, z)
                ex.append((lg.argmax(-1) == yt).all(-1))
        exact_by.append(torch.stack(ex, 1))
    E = torch.cat(exact_by).float()  # [n, nsup]
    res = {f"exact@{s+1}": round(E[:, s].mean().item(), 4) for s in range(a.nsup)}
    res["exact_coarse"] = round(E[M[:, 1] == 1, -1].mean().item(), 4)
    res["exact_fine"] = round(E[M[:, 1] == 0, -1].mean().item(), 4)
    m.train()
    return res

def lr_at(s):
    if s < a.warmup: return a.lr * s / a.warmup
    return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))))

B = a.batch
x, yt = draw(B)
with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
    y, z = model.init_state(B, N * N, dev, torch.float32)
depth = torch.zeros(B, dtype=torch.long, device=dev)
log = open(out / "log.jsonl", "a")
t0, tl = time.time(), time.time()
acc = {"loss": 0.0, "qloss": 0.0, "exact": 0.0, "cell": 0.0, "k": 0}
for step in range(step0, a.steps):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
        h = model.embed(x)
        y, z, lg, q = model.step(h, y, z)
        loss = F.cross_entropy(lg.float().reshape(-1, 2), yt.reshape(-1))
        exact = (lg.argmax(-1) == yt).all(-1)
        qloss = F.binary_cross_entropy_with_logits(q.float(), exact.float())
    opt.zero_grad(set_to_none=True)
    (loss + 0.5 * qloss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - a.ema)
        depth += 1
        stop = (depth >= a.nsup) | ((q > 0) & (torch.rand(B, device=dev) > a.explore))
        if stop.any():
            k = int(stop.sum()); nx, ny = draw(k)
            x, yt = x.clone(), yt.clone(); x[stop], yt[stop] = nx, ny
            y0, z0 = model.init_state(k, N * N, dev, y.dtype)
            y, z = y.clone(), z.clone(); y[stop], z[stop] = y0, z0
            depth[stop] = 0
    acc["loss"] += loss.item(); acc["qloss"] += qloss.item(); acc["exact"] += exact.float().mean().item()
    acc["cell"] += (lg.argmax(-1) == yt).float().mean().item(); acc["k"] += 1
    if (step + 1) % 100 == 0:
        k = acc.pop("k"); r = {kk: round(v / k, 4) for kk, v in acc.items()}
        r.update(step=step + 1, lr=round(lr_at(step), 6), its=round(100 / (time.time() - tl), 2), mean_depth=round(depth.float().mean().item(), 2))
        print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        acc = {"loss": 0.0, "qloss": 0.0, "exact": 0.0, "cell": 0.0, "k": 0}; tl = time.time()
    last = step + 1 == a.steps or (time.time() - t0) / 3600 > a.hours
    if (step + 1) % a.eval_every == 0 or last:
        r = {"step": step + 1, "val": evaluate(ema, Xva, Yva, Mva, a.eval_n), "test_E1M9": evaluate(ema, Xte, Yte, Mte, a.eval_n),
             "elapsed_min": round((time.time() - t0) / 60, 1)}
        print("EVAL " + json.dumps(r), flush=True); log.write(json.dumps({"eval": r}) + "\n"); log.flush()
        ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step + 1, "args": vars(a)}
        torch.save(ck, out / "ckpt.pt.tmp"); os.replace(out / "ckpt.pt.tmp", out / "ckpt.pt")
        torch.save({"ema": ema.state_dict(), "args": vars(a), "step": step + 1}, out / "ema.pt")
    if last: break
print("DONE", flush=True)
