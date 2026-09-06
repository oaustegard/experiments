"""Train a query head at the chosen layer k on cached hidden states.

--head mlp  (default): the phase-1 head.  Reads ONE vector, the layer-k
            residual at the LAST prompt token, and recovers operator + 12
            digit slots.  Recovers the operator at 1.00 and the full query at
            6-12%: the operand digits are not assembled at that position.

--head attn: cross-attention readout over the layer-k hiddens of ALL prompt
            positions (model_utils.AttnQueryHead), which is where the digits
            actually live.  Needs the per-position cache
            data/hidden_seq_<model>_k<k>_<split>.npy (gitignored, regenerable).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import data as D
import model_utils as mu
import probe as P


# ------------------------------------------------------------------ mlp head
def load_layer(model_name, split, k, n=None):
    arr, rows = P.get_hidden(model_name, split, n)
    x = torch.from_numpy(np.ascontiguousarray(arr[:, k + 1, :])).float()
    return x, rows


# ------------------------------------------------------------------ attn head
def seq_cache_path(model_name, k, split):
    return os.path.join(D.data_dir(), f"hidden_seq_{model_name}_k{k}_{split}.npy")


def seq_mask_path(model_name, k, split):
    return os.path.join(D.data_dir(),
                        f"hidden_seq_{model_name}_k{k}_{split}_mask.npy")


def get_hidden_seq(model_name, split, k, n=None, model=None, tok=None,
                   force=False, verbose=True):
    """([N, T, H] fp16 memmap, [N, T] uint8 mask, rows) -- cached on disk."""
    hp = seq_cache_path(model_name, k, split)
    mp = seq_mask_path(model_name, k, split)
    rows = D.load_split(split, n)
    if os.path.exists(hp) and os.path.exists(mp) and not force:
        h = np.load(hp, mmap_mode="r")
        if h.shape[0] >= len(rows):
            return h[:len(rows)], np.load(mp, mmap_mode="r")[:len(rows)], rows
    if model is None:
        model, tok = mu.load_model(model_name)
    h, m = mu.extract_hidden_seq(model, tok, [r["prompt"] for r in rows], k,
                                 verbose=verbose)
    mu.ensure_dir(D.data_dir())
    np.save(hp, h.numpy())
    np.save(mp, m.numpy())
    return np.load(hp, mmap_mode="r"), np.load(mp, mmap_mode="r"), rows


def _take(h, m, idx):
    x = torch.from_numpy(np.ascontiguousarray(h[idx])).float()
    mk = torch.from_numpy(np.ascontiguousarray(m[idx]))
    return x, mk


def evaluate_attn(head, h, m, op_t, slot_t, bs=256):
    ops, slots = [], []
    head.eval()
    with torch.no_grad():
        for i in range(0, h.shape[0], bs):
            idx = slice(i, min(i + bs, h.shape[0]))
            x, mk = _take(h, m, idx)
            o, s = head(x, mk)
            ops.append(o.argmax(-1))
            slots.append(s.argmax(-1))
    head.train()
    ops, slots = torch.cat(ops), torch.cat(slots)
    return {"op_acc": float((ops == op_t).float().mean()),
            "slot_acc": float((slots == slot_t).float().mean()),
            "exact": float(((ops == op_t) & (slots == slot_t).all(-1))
                           .float().mean()),
            "per_slot": (slots == slot_t).float().mean(0).tolist()}


def fit_attn(h, m, op_t, slot_t, hidden, steps=3000, bs=64, lr=1e-3, seed=0,
             verbose=True):
    torch.manual_seed(seed)
    head = mu.AttnQueryHead(hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = h.shape[0]
    g = torch.Generator().manual_seed(seed)
    t0, run = time.time(), 0.0
    for step in range(steps):
        # sorted indices: h is a memmap, so a monotone gather is much cheaper
        idx = torch.randint(0, n, (min(bs, n),), generator=g).sort().values
        x, mk = _take(h, m, idx.numpy())
        o, s = head(x, mk)
        loss = mu.query_loss(o, s, op_t[idx], slot_t[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        run += float(loss)
        if verbose and (step + 1) % 200 == 0:
            print(f"  step {step + 1}/{steps} loss {run / 200:.4f} "
                  f"({(time.time() - t0) / (step + 1):.3f}s/step)", flush=True)
            run = 0.0
    return head, (time.time() - t0) / max(steps, 1)


def head_suffix(head):
    return "" if head == "mlp" else f"_{head}"


def ckpt_file(model_name, head):
    return os.path.join(mu.repo_dir(), "ckpt",
                        f"query_head_{model_name}{head_suffix(head)}.pt")


def results_file(model_name, head):
    return os.path.join(mu.repo_dir(), "results",
                        f"query_head_{model_name}{head_suffix(head)}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mu.MODELS))
    ap.add_argument("--head", default="mlp", choices=["mlp", "attn"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.k is None:
        with open(os.path.join(mu.repo_dir(), "results",
                               f"probe_{args.model}.json")) as f:
            args.k = json.load(f)["chosen_k"]
    if args.bs is None:
        args.bs = 64 if args.head == "attn" else 256

    res = {"model": args.model, "head": args.head, "k": args.k,
           "steps": args.steps, "bs": args.bs}

    if args.head == "mlp":
        x_tr, tr_rows = load_layer(args.model, "train", args.k, args.n_train)
        op_tr, slot_tr = P.targets(tr_rows)
        mean = x_tr.mean(0, keepdim=True)
        std = x_tr.std(0, keepdim=True) + 1e-5
        x_tr = (x_tr - mean) / std
        head = P.fit_head(x_tr, op_tr, slot_tr, x_tr.shape[1], steps=args.steps,
                          bs=args.bs, lr=args.lr, linear=False)
        res["n_train"] = x_tr.shape[0]
        res["params"] = mu.count_params(head)
        for split in ("val", "test_in", "test_len5"):
            x, rows = load_layer(args.model, split, args.k, args.n_eval)
            op, slot = P.targets(rows)
            res[split] = P.evaluate(head, (x - mean) / std, op, slot)
            print(split, {kk: round(vv, 4) for kk, vv in res[split].items()
                          if kk != "per_slot"}, flush=True)
        payload = {"state_dict": head.state_dict(), "k": args.k, "head": "mlp",
                   "mean": mean, "std": std}
    else:
        model, tok = mu.load_model(args.model)
        h_tr, m_tr, tr_rows = get_hidden_seq(args.model, "train", args.k,
                                             args.n_train, model, tok)
        op_tr, slot_tr = P.targets(tr_rows)
        head, sec = fit_attn(h_tr, m_tr, op_tr, slot_tr, h_tr.shape[2],
                             steps=args.steps, bs=args.bs, lr=args.lr)
        res["n_train"] = h_tr.shape[0]
        res["params"] = mu.count_params(head)
        res["sec_per_step"] = sec
        assert res["params"] < 1_500_000, res["params"]
        for split in ("val", "test_in", "test_len5"):
            h, m, rows = get_hidden_seq(args.model, split, args.k, args.n_eval,
                                        model, tok)
            op, slot = P.targets(rows)
            res[split] = evaluate_attn(head, h, m, op, slot)
            print(split, {kk: round(vv, 4) for kk, vv in res[split].items()
                          if kk != "per_slot"}, flush=True)
        payload = {"state_dict": head.state_dict(), "k": args.k, "head": "attn"}

    mu.ensure_dir(os.path.join(mu.repo_dir(), "ckpt"))
    torch.save(payload, ckpt_file(args.model, args.head))
    out = results_file(args.model, args.head)
    mu.ensure_dir(os.path.dirname(out))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"DONE query_head {args.model} {args.head}")


if __name__ == "__main__":
    main()
