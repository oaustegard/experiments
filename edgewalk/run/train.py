"""Distil Jev's sandwich verdicts into a small network the page runs in plain JavaScript.

Input: one one-hot block per piece. Network: 2 tanh hidden layers of 32, one logit; width, weight decay
and epochs chosen by a held-out sweep (results/sweep.json).
Loss: binary cross-entropy against Jev's probability. 15% of cases are held out by hash.
Writes web/sandwich.json (pieces + weights + starting cases) and results/train.json (fit metrics).

    python3 edgewalk/run/train.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sandwich  # noqa: E402

ROOT = HERE.parent
LABELS = ROOT / "data" / "labels.jsonl"
WEB = ROOT / "web" / "sandwich.json"
RES = ROOT / "results" / "train.json"
H = 32
EPOCHS = 80
ZONES = [(0.05, "edge"), (0.15, "fence"), (0.35, "slip"), (9, "fall")]


def zone(p):
    d = abs(p - 0.5)
    return next(z for t, z in ZONES if d <= t)


def encode(case):
    x = []
    for k in sandwich.KEYS:
        v = [0.0] * len(sandwich.VALUES[k]); v[sandwich.VALUES[k].index(case[k])] = 1.0; x += v
    return x


def held_out(case):
    return int(hashlib.sha1(json.dumps(case, sort_keys=True).encode()).hexdigest(), 16) % 100 < 15


def main():
    rows = [json.loads(l) for l in LABELS.read_text().splitlines()]
    X = torch.tensor([encode(r["case"]) for r in rows])
    y = torch.tensor([r["p"] for r in rows]).clamp(1e-3, 1 - 1e-3)
    te = torch.tensor([held_out(r["case"]) for r in rows])
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(), torch.nn.Linear(H, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=3e-2)
    Xtr, ytr = X[~te], y[~te]
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            b = perm[i:i + 128]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr[b]).squeeze(1), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(X).squeeze(1))

    def metrics(mask):
        a, b = p[mask].numpy(), y[mask].numpy()
        za, zb = [zone(v) for v in a], [zone(v) for v in b]
        return {"n": int(mask.sum()), "mae": float(np.abs(a - b).mean()), "pearson": float(np.corrcoef(a, b)[0, 1]),
                "zone_agreement": float(np.mean([u == v for u, v in zip(za, zb)])),
                "jev_in_fence": float(np.mean(np.abs(b - 0.5) <= 0.15))}
    m = {"train": metrics(~te), "test": metrics(te), "cases": len(rows), "hidden": H, "epochs": EPOCHS}
    # single-move swings on the test set: how much one piece change moves the student
    swings = []
    for r in [rows[i] for i in torch.nonzero(te).squeeze(1).tolist()[:200]]:
        base = r["case"]
        with torch.no_grad():
            p0 = torch.sigmoid(net(torch.tensor([encode(base)]))).item()
            alts = []
            for k in sandwich.KEYS:
                for v in sandwich.VALUES[k]:
                    if v != base[k]:
                        c = dict(base); c[k] = v; alts.append(encode(c))
            pa = torch.sigmoid(net(torch.tensor(alts))).squeeze(1).numpy()
        swings += list(np.abs(pa - p0))
    sw = np.array(swings)
    m["single_move_abs_change"] = {"median": float(np.median(sw)), "p75": float(np.percentile(sw, 75)), "p90": float(np.percentile(sw, 90)),
                                   "over_0.10": float((sw > 0.10).mean()), "over_0.20": float((sw > 0.20).mean())}

    # starting cases: a well-known food with at most two pieces changed, which Jev and the student both put near 50/50
    def nearest(case):
        best = min(sandwich.CLASSICS.items(), key=lambda kv: sum(case[k] != v for k, v in zip(sandwich.KEYS, kv[1])))
        return best[0], sum(case[k] != v for k, v in zip(sandwich.KEYS, best[1]))
    starts = []
    for r, pp in zip(rows, p.tolist()):
        name, n = nearest(r["case"])
        if n <= 2 and abs(r["p"] - 0.5) <= 0.1 and abs(pp - 0.5) <= 0.07:
            starts.append({"food": r["case"], "from": name, "changes": n})
    m["starts"] = len(starts)
    m["starts_by_classic"] = {n: sum(s["from"] == n for s in starts) for n in sandwich.CLASSICS}
    W = [l for l in net if isinstance(l, torch.nn.Linear)]
    out = {
        "question": "Is it a sandwich?",
        "teacher": "Jev (TypeSafe), noul: \"This food is a sandwich.\"",
        "pieces": [{"key": k, "label": lbl, "values": vals, "ordered": k in sandwich.ORDERED} for k, lbl, vals in sandwich.PIECES],
        "classics": {n: dict(zip(sandwich.KEYS, v)) for n, v in sandwich.CLASSICS.items()},
        "layers": [{"w": np.round(l.weight.detach().numpy(), 5).tolist(), "b": np.round(l.bias.detach().numpy(), 5).tolist()} for l in W],
        "starts": starts,
    }
    print(json.dumps(m, indent=1))
    RES.parent.mkdir(exist_ok=True); RES.write_text(json.dumps(m, indent=1))
    WEB.write_text(json.dumps(out, separators=(",", ":")))
    print(len(starts), "starts;", WEB.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
