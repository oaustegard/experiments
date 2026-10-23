"""Codebook fine-tuning: a full 262,144-row table stored as tiered product-quantization codes,
whose codebooks (not codes) are trained through the frozen k=4 student.

Tiers (row sets, each with its own M and its own 256-entry codebooks): the 41,741 kept rows,
other rows seen in training-side text (.work/freq.npy > 0), and rows never seen. Codes come from
k-means PQ on the float table (.work/table_<src>.npy with the trained kept rows). Then the
codebooks train on two losses: 1 - cos to EG2 on training texts tokenized with the full
tokenizer (Bluesky training targets + caption training texts), and the one-token loss of
run/tok_rows.py on rows outside the kept set, so the never-seen tier still gets a signal.

Usage: python3 -I run/cbft.py --tiers kept:128 seen:32 unseen:8 --steps 1500 --caption --tag NAME
Writes .work/cb_<tag>.npz (codes + codebooks), results/cbft_<tag>.json (bench before / after).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, HID, LS, QUERY, WORK, Benches, Teacher, cut_rows, cut_student, full_student, tokenize  # noqa: E402
from tcodecs import pq_apply, pq_fit  # noqa: E402

sys.path.insert(0, str(LS / "run"))
from l1_distill import SETS, epoch_batches, train_texts  # noqa: E402
from model import batches  # noqa: E402

torch.set_num_threads(4)
N = 262144
SD = LS / ".work/mac/out/l1/mac_k4_v32k/final.pt"


class PQTable(torch.nn.Module):
    def __init__(self, tier_rows, Ms, X, w, seed=0):
        super().__init__()
        self.tier = torch.full((N,), -1, dtype=torch.long)
        self.local = torch.zeros(N, dtype=torch.long)
        self.Ms = Ms
        self.cbs = torch.nn.ParameterList()
        self.codes = []
        self.fixed = {}
        self.extra_bytes = 0
        for t, (rows, M) in enumerate(zip(tier_rows, Ms)):
            self.tier[torch.from_numpy(rows)] = t
            self.local[torch.from_numpy(rows)] = torch.arange(len(rows))
            if isinstance(M, str):  # "remexB": fixed rows, rotation + Lloyd-Max + fp16 norm per row
                import remex
                b = int(M[5:])
                q = remex.Quantizer(HID, bits=b, seed=0, rotation="haar")
                self.fixed[t] = torch.from_numpy(q.decode(q.encode(X[rows])).astype(np.float32))
                self.cbs.append(torch.nn.Parameter(torch.zeros(1), requires_grad=False))
                self.codes.append(torch.zeros((len(rows), 0), dtype=torch.long))
                self.extra_bytes += len(rows) * (HID * b // 8 + 2)
                continue
            pq = pq_fit(X[rows], M, w=w[rows], seed=seed + t)
            _, codes = pq_apply(pq, X[rows])
            cb = faiss_centroids(pq)
            self.cbs.append(torch.nn.Parameter(torch.from_numpy(cb)))
            self.codes.append(torch.from_numpy(codes.astype(np.int64)))
            self.tier[torch.from_numpy(rows)] = t
            self.local[torch.from_numpy(rows)] = torch.arange(len(rows))

    def rows(self, ids):
        """ids: 1-D long -> (len, 512)."""
        out = torch.zeros(len(ids), HID)
        tier = self.tier[ids]
        for t, M in enumerate(self.Ms):
            m = tier == t
            if not m.any():
                continue
            if t in self.fixed:
                out[m] = self.fixed[t][self.local[ids[m]]]
                continue
            c = self.codes[t][self.local[ids[m]]]  # (n, M)
            cb = self.cbs[t]  # (M, 256, d)
            g = cb[torch.arange(M)[None, :], c]  # (n, M, d)
            out[m] = g.reshape(len(c), HID)
        return out

    def full(self):
        with torch.no_grad():
            return torch.cat([self.rows(torch.arange(s, min(N, s + 16384))) for s in range(0, N, 16384)]).numpy()

    def nbytes(self):
        return self.extra_bytes + sum(len(c) * M for t, (c, M) in enumerate(zip(self.codes, self.Ms)) if t not in self.fixed) \
            + sum(cb.numel() * 2 for t, cb in enumerate(self.cbs) if t not in self.fixed)


def faiss_centroids(pq):
    import faiss
    return faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["kept:128", "seen:32", "unseen:8"],
                    help="set:M in order; sets: keptN (N most frequent kept rows), kept (the rest of the kept rows), seen, unseen")
    ap.add_argument("--src", default="ridge_e3")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tok-batch", type=int, default=256)
    ap.add_argument("--tok-weight", type=float, default=0.5)
    ap.add_argument("--caption", action="store_true", help="also train on train-split alt + post texts")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sd = torch.load(SD, map_location="cpu")
    rows = cut_rows()
    X = np.load(WORK / f"table_{a.src}.npy")
    X[rows] = sd["tm.embed_tokens.weight"].numpy()
    f = np.load(WORK / "freq.npy")
    kept = np.zeros(N, bool)
    kept[rows] = True
    tiers, Ms, taken = [], [], np.zeros(N, bool)
    for spec in a.tiers:
        name, M = spec.split(":")
        if name.startswith("kept") and name != "kept":
            k = int(name[4:])
            r = rows[np.argsort(-f[rows], kind="stable")[:k]]
        else:
            r = {"kept": rows, "seen": np.where(~kept & (f > 0))[0], "unseen": np.where(~kept & (f == 0))[0]}[name]
        r = np.sort(r[~taken[r]])
        taken[r] = True
        tiers.append(r)
        Ms.append(M if M.startswith("remex") else int(M))
    assert taken.all()
    w = f.astype(np.float64) + 1.0
    t0 = time.time()
    tab = PQTable(tiers, Ms, X, w)
    print("pq fit", f"{time.time() - t0:.0f}s", "MB", tab.nbytes() / 1e6, flush=True)

    B = Benches()
    res = {"args": vars(a), "table_mb": tab.nbytes() / 1e6, "tier_sizes": [len(t) for t in tiers]}
    res["before"] = B.run(full_student(sd, table=tab.full()), "full")
    print("before", res["before"], flush=True)

    s = cut_student(sd)
    tm = s.tm
    for p in s.parameters():
        p.requires_grad_(False)
    T = Teacher(threads=1)
    texts, Y = train_texts(SETS)
    if a.caption:
        sys.path.insert(0, str(LS / "run"))
        from img_common import IMG
        for n in ("train_alt", "train_post", "extra_alt", "extra_alt2"):
            p = IMG / f"teacher_{n}.npy"
            if p.exists():
                texts += json.loads((IMG / f"teacher_{n}.json").read_text())
                Y = np.concatenate([Y, np.load(p)])
    enc = tokenize(T, texts)
    lens = np.array([len(e) for e in enc])
    Yt = torch.from_numpy(np.asarray(Y, np.float32))
    V = torch.from_numpy(np.load(WORK / "tok_teacher.npy").astype(np.float32))
    tok_ids = np.where(~kept & (np.linalg.norm(V.numpy(), axis=1) > 0.5))[0]
    pre = T.tok.encode(QUERY).ids
    head, eos = torch.tensor(pre[:-2]), torch.tensor(pre[-1:])
    opt = torch.optim.Adam([p for p in tab.parameters() if p.requires_grad], lr=a.lr)
    scale = HID ** 0.5
    order, ep = epoch_batches(lens, a.batch, 0), 0
    t0 = time.time()

    def fwd(ids, mask):
        u, inv = torch.unique(ids, return_inverse=True)
        x = tab.rows(u)[inv] * scale
        h = tm(inputs_embeds=x, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    for step in range(a.steps):
        if not order:
            ep += 1
            order = epoch_batches(lens, a.batch, ep)
        bi = order.pop()
        _, ids, mask = next(batches([enc[i] for i in bi], len(bi), order=np.arange(len(bi))))
        loss_t = 1 - torch.nn.functional.cosine_similarity(fwd(ids, mask), Yt[bi], dim=-1).mean()
        loss = loss_t
        if a.tok_weight > 0:
            tb = torch.from_numpy(rng.choice(tok_ids, a.tok_batch, replace=False))
            tids = torch.cat([head.expand(len(tb), -1), tb[:, None], eos.expand(len(tb), -1)], 1)
            loss_o = 1 - torch.nn.functional.cosine_similarity(fwd(tids, torch.ones_like(tids)), V[tb], dim=-1).mean()
            loss = loss + a.tok_weight * loss_o
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(step, round(loss_t.item(), 4), round(loss.item(), 4), f"{time.time() - t0:.0f}s", flush=True)
    full = tab.full()
    np.savez(WORK / f"cb_{a.tag}.npz", **{f"codes{t}": c.numpy().astype(np.uint8) for t, c in enumerate(tab.codes)},
             **{f"cb{t}": cb.detach().numpy() for t, cb in enumerate(tab.cbs)}, **{f"fixed{t}": v.numpy() for t, v in tab.fixed.items()}, **{f"rows{t}": r for t, r in enumerate(tiers)})
    res["after"] = B.run(full_student(sd, table=full), "full")
    print("after", res["after"], flush=True)
    (HERE / "results" / f"cbft_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
