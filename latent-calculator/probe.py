"""Per-layer LINEAR probe: where is the arithmetic query readable?

Fits a plain linear map hidden[t] -> (operator, 12 digit slots) at every layer
(k = -1 is the embedding output, k = j is the output of decoder layer j) and
reports operator accuracy, mean per-slot digit accuracy and EXACT query
accuracy (operator plus all 12 slots) on val (in-distribution) and test_len5.

Picks k = the shallowest layer with exact val accuracy > 0.95; if no layer
clears the bar, falls back to the argmax layer and flags it in probe.json.
Hidden states are cached to data/hidden_<model>_<split>.npy (gitignored) so
query_head.py can reuse them.
"""

import argparse
import json
import os

import numpy as np
import torch

import data as D
import model_utils as mu

EXACT_THRESHOLD = 0.95


def cache_path(model_name, split):
    return os.path.join(D.data_dir(), f"hidden_{model_name}_{split}.npy")


def get_hidden(model_name, split, n=None, model=None, tok=None, force=False):
    """[N, L+1, H] fp16 numpy array (memory-mapped), cached on disk."""
    path = cache_path(model_name, split)
    rows = D.load_split(split, n)
    if os.path.exists(path) and not force:
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] >= len(rows):
            return arr[:len(rows)], rows
    if model is None:
        model, tok = mu.load_model(model_name)
    h = mu.extract_hidden(model, tok, [r["prompt"] for r in rows], verbose=True)
    np.save(path, h.numpy())
    return np.load(path, mmap_mode="r"), rows


def targets(rows):
    op = torch.tensor([D.OPS.index(r["op"]) for r in rows])
    slots = torch.tensor([
        D.digits_right_aligned(str(r["a"]), D.N_OPERAND_SLOTS)
        + D.digits_right_aligned(str(r["b"]), D.N_OPERAND_SLOTS) for r in rows])
    return op, slots


def evaluate(head, x, op_t, slot_t, bs=1024):
    ops, slots = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            o, s = head(x[i:i + bs])
            ops.append(o.argmax(-1))
            slots.append(s.argmax(-1))
    ops = torch.cat(ops)
    slots = torch.cat(slots)
    op_acc = float((ops == op_t).float().mean())
    slot_acc = float((slots == slot_t).float().mean())
    exact = float(((ops == op_t) & (slots == slot_t).all(-1)).float().mean())
    per_slot = (slots == slot_t).float().mean(0).tolist()
    return {"op_acc": op_acc, "slot_acc": slot_acc, "exact": exact,
            "per_slot": per_slot}


def fit_head(x, op_t, slot_t, hidden, steps=400, bs=256, lr=1e-2, linear=True,
             mlp=512, seed=0):
    torch.manual_seed(seed)
    head = mu.QueryHead(hidden, mlp=mlp, linear=linear)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        o, s = head(x[idx])
        loss = mu.query_loss(o, s, op_t[idx], slot_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mu.MODELS))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-cache", type=int, default=None,
                    help="rows of train to extract/cache (default: all)")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = mu.load_model(args.model)
    tr_h, tr_rows = get_hidden(args.model, "train", args.n_cache, model, tok)
    va_h, va_rows = get_hidden(args.model, "val", args.n_eval, model, tok)
    t5_h, t5_rows = get_hidden(args.model, "test_len5", args.n_eval, model, tok)
    n_tr = min(args.n_train, tr_h.shape[0])
    tr_h = tr_h[:n_tr]
    tr_rows = tr_rows[:n_tr]

    tr_op, tr_slot = targets(tr_rows)
    va_op, va_slot = targets(va_rows)
    t5_op, t5_slot = targets(t5_rows)
    n_lay = tr_h.shape[1]
    hidden = tr_h.shape[2]

    layers = []
    for j in range(n_lay):
        k = j - 1
        x_tr = torch.from_numpy(np.ascontiguousarray(tr_h[:, j, :])).float()
        mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-5
        head = fit_head((x_tr - mean) / std, tr_op, tr_slot, hidden,
                        steps=args.steps)
        x_va = (torch.from_numpy(np.ascontiguousarray(va_h[:, j, :])).float() - mean) / std
        x_t5 = (torch.from_numpy(np.ascontiguousarray(t5_h[:, j, :])).float() - mean) / std
        rec = {"k": k, "val": evaluate(head, x_va, va_op, va_slot),
               "test_len5": evaluate(head, x_t5, t5_op, t5_slot)}
        layers.append(rec)
        print(f"k={k:3d} val exact={rec['val']['exact']:.3f} "
              f"op={rec['val']['op_acc']:.3f} slot={rec['val']['slot_acc']:.3f} | "
              f"len5 exact={rec['test_len5']['exact']:.3f}", flush=True)

    good = [r for r in layers if r["val"]["exact"] > EXACT_THRESHOLD]
    if good:
        chosen, fallback = good[0]["k"], False
    else:
        chosen = max(layers, key=lambda r: r["val"]["exact"])["k"]
        fallback = True
    out = {"model": args.model, "n_train": n_tr, "steps": args.steps,
           "threshold": EXACT_THRESHOLD, "chosen_k": chosen,
           "fallback": fallback, "layers": layers}
    path = args.out or os.path.join(mu.repo_dir(), "results",
                                    f"probe_{args.model}.json")
    mu.ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"chosen_k={chosen} fallback={fallback} -> {path}")
    print(f"DONE probe {args.model}")


if __name__ == "__main__":
    main()
