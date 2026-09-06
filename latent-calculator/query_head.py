"""Train the 2-layer MLP query head at the chosen layer k (cached hiddens)."""

import argparse
import json
import os

import numpy as np
import torch

import model_utils as mu
import probe as P


def load_layer(model_name, split, k, n=None):
    arr, rows = P.get_hidden(model_name, split, n)
    x = torch.from_numpy(np.ascontiguousarray(arr[:, k + 1, :])).float()
    return x, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mu.MODELS))
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.k is None:
        with open(os.path.join(mu.repo_dir(), "results",
                               f"probe_{args.model}.json")) as f:
            args.k = json.load(f)["chosen_k"]

    x_tr, tr_rows = load_layer(args.model, "train", args.k, args.n_train)
    op_tr, slot_tr = P.targets(tr_rows)
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-5
    x_tr = (x_tr - mean) / std
    head = P.fit_head(x_tr, op_tr, slot_tr, x_tr.shape[1], steps=args.steps,
                      bs=args.bs, lr=args.lr, linear=False)

    res = {"model": args.model, "k": args.k, "n_train": x_tr.shape[0],
           "steps": args.steps, "params": mu.count_params(head)}
    for split in ("val", "test_in", "test_len5"):
        x, rows = load_layer(args.model, split, args.k, args.n_eval)
        op, slot = P.targets(rows)
        res[split] = P.evaluate(head, (x - mean) / std, op, slot)
        print(split, {kk: round(vv, 4) for kk, vv in res[split].items()
                      if kk != "per_slot"}, flush=True)

    ck = os.path.join(mu.repo_dir(), "ckpt")
    mu.ensure_dir(ck)
    torch.save({"state_dict": head.state_dict(), "k": args.k, "mean": mean,
                "std": std}, os.path.join(ck, f"query_head_{args.model}.pt"))
    out = os.path.join(mu.repo_dir(), "results", f"query_head_{args.model}.json")
    mu.ensure_dir(os.path.dirname(out))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"DONE query_head {args.model}")


if __name__ == "__main__":
    main()
