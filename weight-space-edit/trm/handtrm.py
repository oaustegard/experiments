"""Hand-written weights for the trm-doom TRM: exact BFS shortest-route unions, no training.

Same architecture and tensor shapes as the shipped planner (conv mixer, d=128, two blocks, n=4, T=3),
so the weights load into trm-doom/web/trm.js unchanged. Each call of the shared network does one step
of a cellular automaton:

  * Lee's wavefront from GOAL: an unlabelled passable cell next to a cell labelled k takes label k+1 (mod 3).
    The goal starts at label 0, so a cell's label is its BFS distance to the goal, mod 3.
  * Descent marks from START: a labelled cell is marked when it is START or when a neighbour one step
    farther from the goal is marked. Distances of neighbouring cells differ by at most 1, so the mod-3
    labels tell "one farther" apart from "same" and "one nearer". The marked cells are exactly the cells on
    shortest START->GOAL routes; nothing is marked when the goal can't be reached.

Representation. Channel 0 holds a large constant (BIG before normalisation); every other feature is a 0/1
bit at 1 unit, so RMSNorm only rescales and ratios to channel 0 carry the values. TRM feeds the network
sums (x+y+z, y+z); bits in the sum sit at 1/3..1 of a unit, so block 0 re-quantises every bit (threshold
1/6), and block 1's conv counts labelled and marked neighbours, which its MLP thresholds into the next
state. An MLP hidden unit with a large constant gate is a linear pass-through (silu(G) ~ G); two units
with gain beta give relu(v)-relu(v-1) ~ a clamped step. The state only ever gains bits, so the sum of an
older and a newer state re-quantises to the newer one.
"""
import math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trm-doom" / "train"))
from trm import TRM  # noqa: E402

D, HID, BIG = 128, 512, 1000.0
PAD, WALL, FREE, DOOR, HAZARD, START, GOAL = range(7)
C0, XP, XS, XG = 0, 1, 2, 3
L = [4, 5, 6]       # label = distance to goal mod 3
M = [7, 8, 9]       # marked, by label
GM = 10             # goal is marked: the plan is complete (drives the halting head)
NL = [11, 12, 13]   # scratch: labelled neighbours, by label
NM = [14, 15, 16]   # scratch: marked neighbours, by label
BITS = [XP, XS, XG] + L + M + [GM]
NB4 = [(0, 1), (2, 1), (1, 0), (1, 2)]  # (ky, kx) of the four orthogonal neighbours in a 3x3 kernel

C_NOM = math.sqrt(D)          # channel 0 after RMSNorm (BIG dominates the norm)
A_NOM = C_NOM / BIG           # one unit after RMSNorm
G_PASS, BETA, K = 30.0, 40.0, 2.0


def canonical(bits):
    v = torch.zeros(D, dtype=torch.float64); v[C0] = BIG
    for b in bits: v[b] = 1.0
    return v * math.sqrt(D) / v.norm()


class MLP:
    """Builds gu [2*HID, D] and down [D, HID] one hidden unit at a time."""
    def __init__(s):
        s.g = torch.zeros(HID, D, dtype=torch.float64); s.u = torch.zeros(HID, D, dtype=torch.float64)
        s.down = torch.zeros(D, HID, dtype=torch.float64); s.j = 0

    def unit(s):
        s.j += 1; assert s.j <= HID; return s.j - 1

    def linear(s, out, inp, coef):  # out += coef * h[inp]
        j = s.unit(); s.g[j, C0] = G_PASS / C_NOM; s.u[j, inp] = coef
        s.down[out, j] = 1.0 / (G_PASS * torch.sigmoid(torch.tensor(G_PASS, dtype=torch.float64)).item())

    def step(s, out, terms, theta, k=K):
        """out += 1 unit when sum(w * h[c] in units) > theta. Uses relu(z+.5) - relu(z-.5), z = k(s-theta);
        k is chosen so that every input the rule can see lands at |z| >= 1."""
        for off, sign in ((0.5, 1.0), (-0.5, -1.0)):
            j = s.unit()
            for c, w in terms: s.g[j, c] += BETA * k * w / A_NOM
            s.g[j, C0] += BETA * (off - k * theta) / C_NOM
            s.u[j, C0] = 1.0 / C_NOM
            s.down[out, j] = sign * A_NOM / BETA

    def weights(s):
        return torch.cat([s.g, s.u], 0), s.down


def build():
    sd = {}
    emb = torch.stack([canonical(b) for b in ([], [], [XP], [XP], [XP], [XP, XS], [XP, XG])])
    sd["emb.weight"] = emb
    sd["pos"] = torch.zeros(1024, D, dtype=torch.float64)
    sd["y0"] = canonical([]); sd["z0"] = canonical([])
    head = torch.zeros(2, D, dtype=torch.float64)
    for m in M: head[1, m] = 10.0 / A_NOM
    head[1, C0] = -5.0 / C_NOM
    sd["head.weight"] = head
    q = torch.zeros(1, D, dtype=torch.float64); q[0, GM] = 1000.0 / A_NOM  # mean over cells: 1000/L when the goal is marked
    sd["qhead.weight"] = q; sd["qhead.bias"] = torch.tensor([-1e-3], dtype=torch.float64)

    # block 0: no mixing; re-quantise every bit of the summed input, seed the goal's label
    b0 = MLP()
    for c in BITS:
        b0.linear(c, c, -1.0)
        b0.step(c, [(c, 1.0)] + ([(XG, 1.0)] if c == L[0] else []), 1 / 6, k=12.0)  # bits arrive at 1/3..1 unit
    sd["blocks.0.conv.weight"] = torch.zeros(D, D, 3, 3, dtype=torch.float64)
    sd["blocks.0.gu.weight"], sd["blocks.0.down.weight"] = b0.weights()

    # block 1: count neighbours, then one automaton step
    conv = torch.zeros(D, D, 3, 3, dtype=torch.float64)
    for k in range(3):
        for ky, kx in NB4:
            conv[NL[k], L[k], ky, kx] = 1.0
            conv[NM[k], M[k], ky, kx] = 1.0
    sd["blocks.1.conv.weight"] = conv
    b1 = MLP()
    for c in NL + NM + L + M + [GM]:
        b1.linear(c, c, -1.0)  # clear scratch and the old state; X bits pass through untouched
    for k in range(3):
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        # label k1: already k1, or passable, unlabelled and next to a k-labelled cell
        b1.step(L[k1], [(L[k1], 10), (XP, 4), (NL[k], 1), (L[k], -10), (L[k2], -10)], 4.5)
        # mark k: labelled k and (START, already marked, or next to a marked cell labelled k+1)
        b1.step(M[k], [(L[k], 20), (NM[k1], 1), (XS, 4), (M[k], 30)], 20.5)
    b1.step(GM, [(M[0], 1), (XG, 1)], 1.5)
    sd["blocks.1.conv.weight"] = conv
    sd["blocks.1.gu.weight"], sd["blocks.1.down.weight"] = b1.weights()
    print(f"hidden units used: block0 {b0.j}, block1 {b1.j} of {HID}", file=sys.stderr)
    return {k: v.float() for k, v in sd.items()}


def model_from(sd):
    m = TRM(mixer="conv", d=D, layers=2, n=4, T=3).eval()
    own = m.state_dict()
    own.update({k: v for k, v in sd.items() if k in own})
    m.load_state_dict(own)
    return m


def load_bin(stem):
    import json
    man = json.load(open(stem + ".json")); w = np.fromfile(stem + ".bin", dtype="<f4")
    sd = {}
    for k, t in man["tensors"].items():
        v = torch.from_numpy(w[t["offset"]:t["offset"] + int(np.prod(t["shape"]))].copy())
        if k.endswith("conv.weight"):
            o, i, ky, kx = t["shape"]; v = v.view(o, ky, kx, i).permute(0, 3, 1, 2)
        sd[k] = v.reshape(t["shape"])
    return sd


def export(sd, stem):
    """Write OUT.bin/OUT.json in train/export.py's layout."""
    import json
    order = ["emb.weight", "pos", "y0", "z0", "head.weight", "qhead.weight", "qhead.bias"]
    order += sorted(k for k in sd if k.startswith("blocks."))
    parts, tensors, off = [], {}, 0
    for k in order:
        v = sd[k].float()
        if k.endswith("conv.weight"): v = v.permute(0, 2, 3, 1)
        v = v.numpy().ravel(); tensors[k] = {"offset": off, "shape": list(sd[k].shape)}; parts.append(v); off += v.size
    np.concatenate(parts).astype("<f4").tofile(stem + ".bin")
    cfg = {"d": D, "heads": 4, "layers": 2, "n": 4, "T": 3, "nsup": 16, "mixer": "conv", "L": 1024, "step": "hand"}
    json.dump({"config": cfg, "tensors": tensors}, open(stem + ".json", "w"))


@torch.no_grad()
def run(m, X, n, nsup, batch=64):
    """Exact-match bookkeeping per supervision step. X: [B, n*n] uint8. Returns preds [nsup, B, n*n], halts [nsup, B]."""
    m.set_grid(n, n)
    P, Q = [], []
    for b in range(0, len(X), batch):
        x = torch.from_numpy(X[b:b + batch].astype(np.int64))
        h = m.embed(x); y, z = m.init_state(len(x), n * n, "cpu", h.dtype)
        ps, qs = [], []
        for _ in range(nsup):
            y, z, lg, q = m.step(h, y, z)
            ps.append((lg[..., 1] > lg[..., 0]).to(torch.uint8)); qs.append(q > 0)
        P.append(torch.stack(ps)); Q.append(torch.stack(qs))
    return torch.cat(P, 1).numpy(), torch.cat(Q, 1).numpy()


def bfs_union(g, n):
    """Python port of trm-doom/web/grid.js shortestUnion."""
    from collections import deque
    out = np.zeros(n * n, np.uint8)
    s = np.where(g == START)[0]; t = np.where(g == GOAL)[0]
    if not len(s) or not len(t): return out
    s, t = s[0], t[0]
    def bfs(src):
        d = np.full(n * n, -1); d[src] = 0; q = deque([src])
        while q:
            o = q.popleft(); i, j = o % n, o // n
            for p in (o - 1 if i > 0 else -1, o + 1 if i < n - 1 else -1, o - n if j > 0 else -1, o + n if j < n - 1 else -1):
                if p >= 0 and d[p] < 0 and g[p] >= FREE: d[p] = d[o] + 1; q.append(p)
        return d
    ds = bfs(s)
    if ds[t] < 0: return out
    dt = bfs(t)
    out[(ds >= 0) & (dt >= 0) & (ds + dt == ds[t])] = 1
    return out


if __name__ == "__main__":
    sd = build()
    out = Path(__file__).parent / "weights"; out.mkdir(exist_ok=True)
    torch.save(sd, out / "hand.pt"); export(sd, str(out / "hand"))
    print("wrote", out / "hand.bin")
