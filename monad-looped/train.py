"""Retrofit training: continue training Monad on SYNTH with an Inkling-style loop, or without one (control).

    python3 train.py --data data/train_tok --out ckpt/loopB1 --band 12,20 --kmax 3 --kweights 1,2,1 --tokens 5e8
    python3 train.py --data data/train_tok --out ckpt/control --tokens 5e8          # same data order, no loop

Rows are the chat prompt then `reasoning\\n</think>\\n\\nanswer<|im_end|>` + EOS, loss on the target only, as in
monad-skygest-finetune/train.py. Batches are length-bucketed to a token budget (--batch-tokens). Every step
samples its pass count K from --kweights (index 0 = 1 pass); the gate starts at zero, so step 0 is the base
model whatever K is. The control arm draws the same K sequence and ignores it, so both arms see identical
batches in identical order.

Resumable: --out/state.pt holds model, optimizer, step and RNG; a rerun continues from it. Writes
--out/log.jsonl (one line per --log-every steps) and Monad-format checkpoints at --save-every steps.
Prints 'done -> <out>' when finished.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loopmodel import LoopedMonad  # noqa: E402


def load_data(d):
    d = Path(d)
    ids = np.load(d / "ids.npy", mmap_mode="r")          # int16 token ids, all rows concatenated
    off = np.load(d / "offsets.npy")                      # row starts, len = n_rows + 1
    tgt = np.load(d / "target_start.npy")                 # per row: index (within row) where the loss starts
    return ids, off, tgt


def make_batches(off, batch_tokens, seed, max_len):
    """Length-bucketed batches of row indices, shuffled. Deterministic in seed."""
    lens = np.diff(off)
    rows = np.nonzero(lens <= max_len)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    out = []
    # bucket within chunks of 8192 rows so batches stay shuffled across the corpus
    for c in range(0, len(rows), 8192):
        chunk = rows[c:c + 8192]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        b, L = [], 0
        for r in chunk:
            L2 = max(L, lens[r])
            if b and L2 * (len(b) + 1) > batch_tokens:
                out.append(b)
                b, L2 = [], lens[r]
            b.append(int(r))
            L = L2
        if b:
            out.append(b)
    rng.shuffle(out)
    return out


def collate(ids, off, tgt, rows, pad):
    L = max(int(off[r + 1] - off[r]) for r in rows)
    x = torch.full((len(rows), L), pad, dtype=torch.long)
    m = torch.zeros((len(rows), L), dtype=torch.float32)
    for i, r in enumerate(rows):
        seq = torch.from_numpy(ids[off[r]:off[r + 1]].astype(np.int64))
        x[i, :len(seq)] = seq
        m[i, int(tgt[r]):len(seq)] = 1.0
    return x, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models/monad"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", default="")
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--kweights", default="1,2,1")
    ap.add_argument("--tokens", type=float, default=5e8, help="tokens processed (prompt + target, padding excluded)")
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke runs)")
    ap.add_argument("--init-from", default="", help="Monad-format checkpoint to resume weights from when state.pt is gone")
    ap.add_argument("--init-step", type=int, default=0, help="step that checkpoint was saved at")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    band = tuple(int(x) for x in a.band.split(",")) if a.band else None
    kw = [float(x) for x in a.kweights.split(",")]
    src = a.init_from if (a.init_from and not (out / "state.pt").exists()) else a.model
    model = LoopedMonad.from_pretrained(src, band=band, kmax=a.kmax if band else 1).to(dev)
    model.band, model.kmax = band, (a.kmax if band else 1)
    model.grad_ckpt = a.grad_ckpt
    if a.compile:
        for i, l in enumerate(model.layers):
            model.layers[i] = torch.compile(l, dynamic=True)
    ids, off, tgt = load_data(a.data)
    from transformers import AutoTokenizer
    pad = AutoTokenizer.from_pretrained(a.model).pad_token_id or 0
    batches = make_batches(off, a.batch_tokens, a.seed, a.max_len)
    lens = np.diff(off)
    tok_per_batch = np.mean([lens[b].sum() for b in batches[:2000]])
    total = int(a.tokens / tok_per_batch)
    if a.max_steps:
        total = min(total, a.max_steps)
    epochs_needed = total / len(batches)
    gate_params = [model.gate] if band else []
    base_params = [p for n, p in model.named_parameters() if n != "gate"]
    groups = [{"params": base_params, "lr": a.lr, "weight_decay": 0.01}]
    if gate_params:
        groups.append({"params": gate_params, "lr": a.gate_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=dev == "cuda")
    step = 0
    st = out / "state.pt"
    if st.exists():
        s = torch.load(st, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
        print(f"resumed at step {step}", flush=True)
    elif a.init_from:
        step = a.init_step
        print(f"resumed weights from {a.init_from} at step {step}; optimizer state is fresh (200-step re-warmup)", flush=True)
    rewarm_from = step if (a.init_from and step and not st.exists()) else None
    meta = {"band": band, "kmax": a.kmax, "kweights": kw, "steps": total, "batches_per_epoch": len(batches),
            "epochs": round(epochs_needed, 3), "tok_per_batch": float(tok_per_batch), "lr": a.lr, "gate_lr": a.gate_lr,
            "batch_tokens": a.batch_tokens, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    print(json.dumps(meta), flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    warm = max(1, int(total * a.warmup))
    krng = random.Random(a.seed)
    ks = [krng.choices(range(1, len(kw) + 1), weights=kw)[0] for _ in range(total)]
    model.train()
    t0, tok_seen, agg = time.time(), 0, {}
    while step < total:
        b = batches[step % len(batches)]
        x, m = collate(ids, off, tgt, b, pad)
        x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        K = ks[step] if band else 1
        lr_f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        if rewarm_from is not None:
            lr_f *= min(1.0, (step - rewarm_from + 1) / 200)
        opt.param_groups[0]["lr"] = a.lr * lr_f
        if gate_params:
            opt.param_groups[1]["lr"] = a.gate_lr * lr_f
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(x, K=K)[:, :-1]
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
        mm = m[:, 1:].reshape(-1)
        loss = (nll * mm).sum() / mm.sum().clamp(min=1)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_seen += int(lens[b].sum())
        a_k = agg.setdefault(K, [0.0, 0])
        a_k[0] += loss.item(); a_k[1] += 1
        if step % a.log_every == 0 or step == total:
            dt = time.time() - t0
            rec = {"step": step, "of": total, "lr": round(a.lr * lr_f, 7), "gnorm": round(float(gn), 3),
                   "tok_s": round(tok_seen / dt), "elapsed_s": round(dt),
                   "loss_by_K": {str(k): round(v[0] / v[1], 4) for k, v in sorted(agg.items())}}
            if band:
                rec["gate_mean"] = [round(float(g.mean()), 4) for g in model.gate.detach()]
                rec["gate_absmax"] = [round(float(g.abs().max()), 4) for g in model.gate.detach()]
            print(json.dumps(rec), flush=True)
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            agg = {}
        if step % a.ckpt_every == 0 or step == total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, str(st) + ".tmp")
            Path(str(st) + ".tmp").replace(st)
        if step % a.save_every == 0 or step == total:
            model.save(out / f"step{step}", a.model)
            print(f"saved {out}/step{step}", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
