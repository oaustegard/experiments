#!/usr/bin/env python3
"""Full fine-tune of Monad (56M) on the Bluesky routing task. CPU, fp32.

Loss is computed on the completion only — the tool block is 574 of ~770 tokens
per row, and training the model to reproduce a catalogue it is handed at
inference time would spend most of the gradient on copying.

    python3 train.py --epochs 3 --batch-size 4
    python3 train.py --probe 3          # time three steps and exit

LoRA is not used. At 56M parameters a full fine-tune fits in about a gigabyte of
optimizer state, and the comparison arm (Needle) had no choice — its engine only
accepts LoRA — so full fine-tuning is the honest version of "what you can do
with this model", not a matched condition.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def encode(tok, rows, max_len: int):
    import torch

    ids_all, mask_all = [], []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(r["target"], add_special_tokens=False)["input_ids"]
        ids = [tok.bos_token_id] + p + t + [tok.eos_token_id]
        loss_mask = [0] * (1 + len(p)) + [1] * (len(t) + 1)
        ids, loss_mask = ids[:max_len], loss_mask[:max_len]
        pad = max_len - len(ids)
        ids_all.append(ids + [tok.pad_token_id] * pad)
        mask_all.append(loss_mask + [0] * pad)
    return torch.tensor(ids_all), torch.tensor(mask_all, dtype=torch.float32)


def batch_loss(model, ids, loss_mask, pad_id):
    import torch.nn.functional as F

    attn = (ids != pad_id).long()
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1]
    target = ids[:, 1:]
    m = loss_mask[:, 1:]
    tok_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
    ).view(target.shape)
    denom = m.sum().clamp(min=1.0)
    return (tok_loss * m).sum() / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--model", default=str(HERE / "model"))
    ap.add_argument("--out", default=str(HERE / "tuned"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=896)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--probe", type=int, default=0, help="time N steps and exit")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.train()
    model.config.use_cache = False

    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * a.val_split)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    ids, msk = encode(tok, train_rows, a.max_len)
    vids, vmsk = encode(tok, val_rows, a.max_len)
    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = math.ceil(len(train_rows) / a.batch_size)
    total = steps_per_epoch * a.epochs
    print(
        f"  model     {n_params / 1e6:.1f}M params  seq {a.max_len}  threads {a.threads}\n"
        f"  data      {len(train_rows)} train / {len(val_rows)} val\n"
        f"  schedule  {total} steps ({steps_per_epoch}/epoch)  lr {a.lr}  batch {a.batch_size}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = max(1, total // 20)

    def lr_at(step):
        if step < warmup:
            return a.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * p))

    order = list(range(len(train_rows)))
    step = 0
    t_start = time.perf_counter()
    for epoch in range(a.epochs):
        rng.shuffle(order)
        for i in range(0, len(order), a.batch_size):
            sel = order[i : i + a.batch_size]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = batch_loss(model, ids[sel], msk[sel], tok.pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == 1:
                el = time.perf_counter() - t_start
                print(
                    f"  step      {step}/{total}  loss {loss.item():.4f}  "
                    f"{el / step:.1f}s/step  eta {(total - step) * el / step / 60:.0f}m",
                    flush=True,
                )
            if a.probe and step >= a.probe:
                print(f"  probe     {a.probe} steps in {time.perf_counter() - t_start:.1f}s", flush=True)
                return 0

        model.eval()
        with torch.no_grad():
            vl = [
                batch_loss(model, vids[j : j + a.batch_size], vmsk[j : j + a.batch_size], tok.pad_token_id).item()
                for j in range(0, len(val_rows), a.batch_size)
            ]
        model.train()
        print(f"  epoch     {epoch + 1}/{a.epochs}  val {sum(vl) / len(vl):.4f}", flush=True)

        # Checkpoint every epoch: a 2.3-hour CPU run that only writes at the end
        # has one failure mode too many, and an epoch-1 model is a real arm.
        ck = Path(a.out + f"-e{epoch + 1}")
        ck.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(ck)
        tok.save_pretrained(ck)
        model.config.use_cache = False
        print(f"  saved     {ck}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"  saved     {out}  ({time.perf_counter() - t_start:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
