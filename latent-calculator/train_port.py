"""Train a result encoder against the FROZEN LM (one encoder per injection arm).

Efficiency: layers 0..k are injection-independent, so the layer-k output for the
whole (prompt+answer) sequence is computed ONCE per row and cached in RAM as
fp16; every training step then runs only layers k+1..L via
model_utils.forward_upper (verified bit-identical to a full forward).  RAM cost
is n_rows * seq_len * H * 2 bytes -- 20k * ~24 * 256 * 2 = 0.25 GB for Monad and
0.55 GB for SmolLM2, so the cache stays in memory rather than on disk (a resumed
run simply rebuilds it, which costs one lower-stack pass).

For the residual/delayed arms the injection during training is a direct add into
the cached layer-k hidden at position t (residual) or t+1 (delayed) -- exactly
what the forward hook does at eval time (tested).  The kv arm uses the
latent_slot attention context, since its effect lives inside layer k+1.
"""

import argparse
import json
import os
import time

import torch

import data as D
import model_utils as mu


def journal_path():
    return os.path.join(mu.repo_dir(), "journal.jsonl")


def read_journal(model, arm):
    done = {}
    p = journal_path()
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("model") == model and r.get("arm") == arm:
                    done[r["epoch"]] = r
    return done


def append_journal(rec):
    with open(journal_path(), "a") as f:
        f.write(json.dumps(rec) + "\n")


def ckpt_path(model, arm, epoch):
    return os.path.join(mu.repo_dir(), "ckpt", f"{model}_{arm}_ep{epoch}.pt")


def build_cache(model, tok, rows, k, batch_size=32, verbose=True):
    """Per-row fp16 layer-k hidden states over the full prompt+answer sequence."""
    cache = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            b = mu.encode_rows(tok, chunk)
            out = model(input_ids=b["input_ids"],
                        attention_mask=b["attention_mask"],
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[k + 1]
            for j in range(len(chunk)):
                n = int(b["attention_mask"][j].sum())
                cache.append(h[j, :n].to(torch.float16).clone())
            if verbose and i % (batch_size * 20) == 0:
                print(f"  cache {i}/{len(rows)} ({time.time() - t0:.0f}s)",
                      flush=True)
    return cache


def make_batch(tok, rows, cache, idx, hidden):
    sub = [rows[i] for i in idx]
    b = mu.encode_rows(tok, sub)
    tmax = b["input_ids"].shape[1]
    h = torch.zeros((len(idx), tmax, hidden))
    for j, i in enumerate(idx):
        c = cache[i]
        h[j, :c.shape[0]] = c.float()
    return b, h


def run_batch(model, enc, b, h, k, arm, syms):
    vec = enc(syms)
    if arm in ("residual", "delayed"):
        off = 1 if arm == "delayed" else 0
        tgt = b["t"] + off
        h = h.clone()
        rows = torch.arange(h.shape[0])
        ok = tgt < h.shape[1]
        h[rows[ok], tgt[ok]] = h[rows[ok], tgt[ok]] + vec[ok]
        logits = mu.forward_upper(model, h, b["attention_mask"], k + 1)
    elif arm == "kv":
        ek, ev = mu.make_slot_kv(model, k, vec)
        with mu.injection("kv", k, vec, b["t"], extra_k=ek, extra_v=ev):
            logits = mu.forward_upper(model, h, b["attention_mask"], k + 1)
    else:
        raise ValueError(arm)
    return logits


def evaluate(model, tok, enc, rows, cache, k, arm, hidden, bs=32):
    enc.eval()
    tot, hit, loss_sum, nb = 0, 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            idx = list(range(i, min(i + bs, len(rows))))
            b, h = make_batch(tok, rows, cache, idx, hidden)
            syms = mu.result_symbols([rows[j]["result_string"] for j in idx])
            logits = run_batch(model, enc, b, h, k, arm, syms)
            loss_sum += float(mu.answer_token_loss(logits, b["labels"]))
            nb += 1
            lg = logits[:, :-1, :].argmax(-1)
            lb = b["labels"][:, 1:]
            m = lb != -100
            hit += int((lg[m] == lb[m]).sum())
            tot += int(m.sum())
    enc.train()
    return hit / max(tot, 1), loss_sum / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mu.MODELS))
    ap.add_argument("--arm", required=True, choices=["residual", "kv", "delayed"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.k is None:
        with open(os.path.join(mu.repo_dir(), "results",
                               f"probe_{args.model}.json")) as f:
            args.k = json.load(f)["chosen_k"]
    k = args.k
    torch.manual_seed(args.seed)

    model, tok = mu.load_model(args.model)
    hidden = mu.hidden_size(model)
    enc = mu.ResultEncoder(hidden)
    print(f"encoder params: {mu.count_params(enc)}", flush=True)

    done = read_journal(args.model, args.arm) if args.resume else {}
    start_epoch = 0
    while (start_epoch + 1) in done:
        start_epoch += 1
    if start_epoch:
        p = ckpt_path(args.model, args.arm, start_epoch)
        enc.load_state_dict(torch.load(p)["state_dict"])
        print(f"resumed from {p} (epochs done: {start_epoch})", flush=True)
    if start_epoch >= args.epochs:
        print(f"DONE train_port {args.model} {args.arm}")
        return

    train_rows = D.load_split("train", args.n_train)
    val_rows = D.load_split("val", args.n_val)
    print(f"caching layer-{k} hidden for {len(train_rows)} train rows",
          flush=True)
    tr_cache = build_cache(model, tok, train_rows, k)
    va_cache = build_cache(model, tok, val_rows, k, verbose=False)
    nbytes = sum(c.numel() * 2 for c in tr_cache) / 1e6
    print(f"cache size: {nbytes:.0f} MB", flush=True)

    syms_all = mu.result_symbols([r["result_string"] for r in train_rows])
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    mu.ensure_dir(os.path.join(mu.repo_dir(), "ckpt"))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        perm = torch.randperm(len(train_rows), generator=g).tolist()
        t0 = time.time()
        tot_loss, nsteps = 0.0, 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            b, h = make_batch(tok, train_rows, tr_cache, idx, hidden)
            logits = run_batch(model, enc, b, h, k, args.arm, syms_all[idx])
            loss = mu.answer_token_loss(logits, b["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss)
            nsteps += 1
            if nsteps % 20 == 0:
                print(f"  ep{epoch} step {nsteps} loss {tot_loss / nsteps:.4f} "
                      f"({(time.time() - t0) / nsteps:.2f}s/step)", flush=True)
        wall = time.time() - t0
        acc, vloss = evaluate(model, tok, enc, val_rows, va_cache, k, args.arm,
                              hidden)
        torch.save({"state_dict": enc.state_dict(), "k": k, "arm": args.arm},
                   ckpt_path(args.model, args.arm, epoch))
        torch.save({"state_dict": enc.state_dict(), "k": k, "arm": args.arm},
                   os.path.join(mu.repo_dir(), "ckpt",
                                f"{args.model}_{args.arm}_final.pt"))
        rec = {"model": args.model, "arm": args.arm, "epoch": epoch, "k": k,
               "train_loss": tot_loss / max(nsteps, 1), "val_loss": vloss,
               "val_answer_token_acc": acc, "wall_seconds": wall,
               "steps": nsteps, "sec_per_step": wall / max(nsteps, 1),
               "batch": args.batch, "n_train": len(train_rows)}
        append_journal(rec)
        print(json.dumps(rec), flush=True)

    print(f"DONE train_port {args.model} {args.arm}")


if __name__ == "__main__":
    main()
