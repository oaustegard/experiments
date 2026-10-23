"""Learn token rows the cut never trained, from one-token queries.

The k=4 student's layers stay frozen. For each learned id i the input is the query prompt with
token i in place of its trailing "▁", and the loss is 1 - cos(student, EG2's vector for the same
ids) (.work/tok_teacher.npy). Only rows are optimized (sparse Adam).

Usage: python3 -I run/tok_rows.py --init ridge --epochs 4 --holdout 2000 --tag NAME
  --holdout N   reset N random kept rows to the init and learn them too, then score how close they
                land to their real trained rows and what swapping them in costs on the benchmarks
Writes .work/table_<tag>.npy (full 262,144-row table) and results/tok_rows_<tag>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, HID, LS, QUERY, WORK, Benches, Teacher, cut_rows, cut_student, teacher_table  # noqa: E402

torch.set_num_threads(4)
SD = LS / ".work/mac/out/l1/mac_k4_v32k/final.pt"


def cos(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def ridge_init(E, V, Y, fit_rows, lam=1e-2):
    """Predict trained rows from [EG2 row, one-token EG2 vector, 1]; fit on fit_rows."""
    X = np.hstack([E, V, np.ones((len(E), 1), np.float32)]).astype(np.float64)
    A = np.linalg.solve(X[fit_rows].T @ X[fit_rows] + lam * np.eye(X.shape[1]), X[fit_rows].T @ Y[fit_rows])
    return (X @ A).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="ridge", choices=["ridge", "eg2", "eg2scaled"])
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--learn", default="rare", choices=["rare", "all"], help="rare = ids outside the cut (+ holdout)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args()

    sd = torch.load(SD, map_location="cpu")
    rows = cut_rows()
    N = 262144
    E = np.asarray(teacher_table(), np.float32)
    V = np.load(WORK / "tok_teacher.npy").astype(np.float32)
    trained = np.array(E)
    trained[rows] = sd["tm.embed_tokens.weight"].numpy()
    rng = np.random.default_rng(0)
    hold = np.sort(rng.choice(rows, a.holdout, replace=False)) if a.holdout else np.array([], np.int64)
    fit = np.setdiff1d(rows, hold)
    if a.init == "ridge":
        init = ridge_init(E, V, trained, fit)
    elif a.init == "eg2scaled":
        init = E * np.float32(np.median(np.linalg.norm(trained[fit], axis=1) / np.linalg.norm(E[fit], axis=1)))
    else:
        init = E.copy()
    table = np.array(trained)
    rare = np.setdiff1d(np.arange(N), rows)
    learn = np.concatenate([rare, hold]) if a.learn == "rare" else np.arange(N)
    learn = learn[np.linalg.norm(V[learn], axis=1) > 0.5]  # soft-token placeholders have no target
    rare = rare[np.linalg.norm(V[rare], axis=1) > 0.5]
    table[learn] = init[learn]
    res = {"args": vars(a), "n_learn": int(len(learn))}
    if len(hold):
        res["holdout_init_cos_to_trained"] = float(cos(init[hold], trained[hold]).mean())

    s = cut_student(sd)
    tm = s.tm
    for p in s.parameters():
        p.requires_grad_(False)
    emb = torch.nn.Embedding(N, HID, sparse=True)
    emb.weight.data = torch.from_numpy(table)
    opt = torch.optim.SparseAdam(emb.parameters(), lr=a.lr)
    T = Teacher(threads=1)
    pre = T.tok.encode(QUERY).ids
    head, eos = torch.tensor(pre[:-2]), torch.tensor(pre[-1:])
    tgt = torch.from_numpy(V)
    scale = HID ** 0.5
    steps = int(a.epochs * len(learn) / a.batch)
    t0 = time.time()
    order = rng.permutation(learn)
    pos = 0
    for step in range(steps):
        if pos + a.batch > len(order):
            order, pos = rng.permutation(learn), 0
        b = torch.from_numpy(order[pos:pos + a.batch])
        pos += a.batch
        ids = torch.cat([head.expand(len(b), -1), b[:, None], eos.expand(len(b), -1)], 1)
        x = emb(ids) * scale
        h = tm(inputs_embeds=x, attention_mask=torch.ones_like(ids)).last_hidden_state.mean(1)
        loss = 1 - torch.nn.functional.cosine_similarity(h, tgt[b], dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        # the prompt rows are shared and already trained: keep them fixed
        g = emb.weight.grad.coalesce()
        keep = ~torch.isin(g.indices()[0], torch.cat([head, eos]))
        emb.weight.grad = torch.sparse_coo_tensor(g.indices()[:, keep], g.values()[keep], g.shape)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(step, steps, round(loss.item(), 4), f"{time.time() - t0:.0f}s", flush=True)
    table = emb.weight.data.numpy()
    np.save(WORK / f"table_{a.tag}.npy", table)

    # one-token fidelity on the kept rows with their real trained rows, for reference
    @torch.no_grad()
    def onetok(tab, idx):
        out = []
        for s0 in range(0, len(idx), 2048):
            bb = torch.from_numpy(idx[s0:s0 + 2048])
            ids = torch.cat([head.expand(len(bb), -1), bb[:, None], eos.expand(len(bb), -1)], 1)
            x = torch.from_numpy(tab)[ids] * scale
            h = tm(inputs_embeds=x, attention_mask=torch.ones_like(ids)).last_hidden_state.mean(1)
            out.append(torch.nn.functional.cosine_similarity(h, tgt[bb], dim=-1).numpy())
        return float(np.concatenate(out).mean())

    probe = rng.choice(fit, 4000, replace=False)
    res["onetok_cos_kept_trained"] = onetok(trained, probe)
    res["onetok_cos_rare_learned"] = onetok(table, rng.choice(rare, 4000, replace=False))
    res["onetok_cos_rare_init"] = onetok(np.where(np.isin(np.arange(N), learn)[:, None], init, trained).astype(np.float32),
                                         rng.choice(rare, 4000, replace=False))
    if len(hold):
        res["holdout_learned_cos_to_trained"] = float(cos(table[hold], trained[hold]).mean())
        res["onetok_cos_hold_trained"] = onetok(trained, hold)
        res["onetok_cos_hold_learned"] = onetok(table, hold)
    print(json.dumps(res), flush=True)
    if not a.no_bench:
        B = Benches()
        res["bench"] = {}
        pos_in_cut = np.searchsorted(rows, hold)
        for name, tab in [("cut_true", trained[rows]), ("cut_hold_init", None), ("cut_hold_learned", table[rows])]:
            if tab is None:
                tab = trained[rows].copy()
                tab[pos_in_cut] = init[hold]
            res["bench"][name] = B.run(cut_student(sd, tab), "cut")
            print(name, res["bench"][name], flush=True)
        from vc import full_student
        res["bench"]["full_learned"] = B.run(full_student(sd, table=np.where(np.isin(np.arange(N), hold)[:, None], trained, table).astype(np.float32)), "full")
        print("full_learned", res["bench"]["full_learned"], flush=True)
    (HERE / "results" / f"tok_rows_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
