#!/usr/bin/env python3
"""Fine-tune Gemma 3 270M on the stage-1 rows, under either prompt condition.

`nl2sh-retrieval/gemma_arm.py train` is the stage-1 recipe: 600 NL2Bash rows
whose gold utility has a tldr page, gold plus two distractors as sources, one
epoch, full-parameter fp32 AdamW at 1e-4, loss masked to the model turn. This
keeps every one of those settings and swaps only the user turn, so a fine-tune
on the instantiation prompt is comparable to stage 1's 0.706 rather than to a
new recipe.

Why train both: a zero-shot difference between the prompts measures what the
instruction-tuned model does with an instruction. If instantiation is the better
*framing* of the task, the model trained on it should also end up ahead — and if
it does not, the framing is a prompt-level artifact and worth knowing as one.

    python3 train.py --condition instantiate --tldr <tldr>/pages \
        --nl2bash <nl2bash>/data/bash --out ft_instantiate
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RETRIEVAL = HERE.parent / "nl2sh-retrieval"
sys.path.insert(0, str(RETRIEVAL))
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402
from run_gen import make_sources  # noqa: E402

BASE = "unsloth/gemma-3-270m-it"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(prompts.BUILDERS))
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--nl2bash", type=Path, required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--budget-minutes", type=float, default=60)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pleias_gate as G

    tldr = G.load_tldr(a.tldr)
    nls = (a.nl2bash / "all.nl").read_text(errors="replace").splitlines()
    cms = (a.nl2bash / "all.cm").read_text(errors="replace").splitlines()
    rng = random.Random(a.seed)
    pool = [(n, c) for n, c in zip(nls, cms) if G.gold_utility(c) in tldr]
    rng.shuffle(pool)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32).train()
    build = prompts.BUILDERS[a.condition]

    rows = []
    for nl, cm in pool[: a.rows]:
        srcs = make_sources(G.gold_utility(cm), tldr, rng, a.distractors)
        prompt = tok.apply_chat_template([{"role": "user", "content": build(nl, srcs)}],
                                         tokenize=False, add_generation_prompt=True)
        full = prompt + cm + tok.eos_token
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        fids = tok(full, add_special_tokens=False)["input_ids"][:1024]
        labels = ([-100] * len(pids) + fids[len(pids):])[:1024]
        rows.append({"input_ids": fids, "labels": labels})

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        }

    dl = DataLoader(rows, batch_size=a.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(dl) * a.epochs))

    t0, step, last = time.time(), 0, None
    for ep in range(a.epochs):
        for batch in dl:
            loss = model(**batch).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            step += 1; last = loss.item()
            if step % 25 == 0:
                print(f"  ep{ep} step {step}/{len(dl)*a.epochs} loss {last:.4f} "
                      f"{(time.time()-t0)/60:.1f}m", flush=True)
            if (time.time() - t0) / 60 >= a.budget_minutes:
                print("budget spent; stopping"); break

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    meta = {"base": a.base, "condition": a.condition, "rows": a.rows, "steps": step,
            "final_loss": round(last, 4) if last is not None else None,
            "minutes": round((time.time() - t0) / 60, 1), "seed": a.seed}
    (a.out / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
