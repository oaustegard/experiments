#!/usr/bin/env python3
"""Staged, not run: does fine-tuning install the output shape the gate found missing?

`pleias_gate.py` established that Pleias-RAG-350M zero-shot produces 0 usable
commands out of 40 — it answers with cited encyclopedic prose, because that is
what it was trained to do. The gate deliberately does not answer the next
question, which is whether that is a *shape* problem or a *capability* problem.

There is a direct precedent. `monad-bsky` took this model's 56M sibling from
**0.000 to 0.481** routable with 800 rows, three epochs and 6,466 s on four CPU
cores, loss on completion tokens only. Installing an output shape is exactly
what that fine-tune did. Pleias-RAG-350M is 6.3x the parameters, so the same
recipe scales to roughly eleven hours here — which is why this script takes a
wall-clock budget and checkpoints, rather than assuming it will finish.

Training rows mirror the gate's inference format exactly, so a gain cannot come
from a format change:

    prompt  = <|query_start|>REQUEST<|query_end|>
              <|source_start|><|source_id|>N tldr example<|source_end|>  (x3)
              <|language_start|>...<|query_report_start|>Trivial...<|answer_start|>
    target  = GOLD COMMAND <|answer_end|>

Loss is masked to the target span only. Sources always include the gold
utility's tldr example plus two distractors — the k=3 shortlist the gate found
the model degrades past (6/8 at 3 sources, 4/8 at 5, worse at 15).

    python3 finetune_gate.py --build-only          # write data, print stats, stop
    python3 finetune_gate.py --rows 600 --epochs 1 --budget-minutes 90
    python3 pleias_gate.py --model-path ./ft   # re-run the gate on the result
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_rows(tldr_dir: Path, nl2bash_dir: Path, n: int, distractors: int,
               seed: int) -> list[dict]:
    import pleias_gate as G
    tldr = G.load_tldr(tldr_dir)
    nls = (nl2bash_dir / "all.nl").read_text(encoding="utf-8", errors="replace").splitlines()
    cms = (nl2bash_dir / "all.cm").read_text(encoding="utf-8", errors="replace").splitlines()
    rng = random.Random(seed)
    pool = [(x, c) for x, c in zip(nls, cms) if G.gold_utility(c) in tldr]
    rng.shuffle(pool)
    others = [u for u in tldr if len(tldr[u]) >= 1]

    rows = []
    for nl, cm in pool:
        if len(rows) >= n:
            break
        gu = G.gold_utility(cm)
        picks = [gu] + rng.sample([u for u in others if u != gu], distractors)
        rng.shuffle(picks)
        sources = [f"{u} — {d}: {c}" for u in picks for d, c in tldr[u][:1]]
        rows.append({
            "prompt": G.build_prompt(nl, sources) + G.PREFILL,
            "target": cm + "<|answer_end|>",
            "gold_utility": gu, "nl": nl,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--nl2bash", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--budget-minutes", type=float, default=90.0,
                    help="stop and checkpoint when this wall-clock budget is spent")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", type=Path, default=HERE / "ft")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="save every N steps; a container restart killed a run at "
                         "step 110/300 with nothing on disk, because the first "
                         "version only saved at the end")
    a = ap.parse_args()

    rows = build_rows(a.tldr, a.nl2bash, a.rows + a.holdout, a.distractors, a.seed)
    train, held = rows[a.holdout:], rows[: a.holdout]
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "ft_train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train))
    (HERE / "data" / "ft_holdout.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in held))
    print(f"train {len(train)}  holdout {len(held)}  "
          f"distinct gold utilities {len({r['gold_utility'] for r in rows})}")
    if a.build_only:
        print(f"\nexample prompt:\n{train[0]['prompt'][:400]}\n---\ntarget: {train[0]['target']}")
        return 0

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pleias_gate as G

    tok = AutoTokenizer.from_pretrained(G.MODEL)
    model = AutoModelForCausalLM.from_pretrained(G.MODEL, dtype=torch.float32)
    model.train()

    def encode(r):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(r["target"], add_special_tokens=False)["input_ids"]
        ids = (p + t)[:1024]
        # loss on the completion only, as monad-bsky did
        labels = ([-100] * len(p) + t)[:1024]
        return {"input_ids": ids, "labels": labels}

    enc = [encode(r) for r in train]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        }

    dl = DataLoader(enc, batch_size=a.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(dl) * a.epochs))

    t0, step, stopped = time.time(), 0, False
    for ep in range(a.epochs):
        for batch in dl:
            loss = model(**batch).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                el = (time.time() - t0) / 60
                print(f"  ep{ep} step {step}/{len(dl) * a.epochs}  loss {loss.item():.4f}  "
                      f"{el:.1f}m elapsed", flush=True)
            if a.checkpoint_every and step % a.checkpoint_every == 0:
                a.out.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(a.out); tok.save_pretrained(a.out)
                json.dump({"steps": step, "partial": True,
                           "minutes": round((time.time() - t0) / 60, 1)},
                          open(HERE / "results_finetune_partial.json", "w"), indent=1)
                print(f"  checkpointed at step {step}", flush=True)
            if (time.time() - t0) / 60 >= a.budget_minutes:
                print(f"budget of {a.budget_minutes}m spent at step {step}; checkpointing")
                stopped = True
                break
        if stopped:
            break

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump({"steps": step, "epochs_requested": a.epochs, "rows": len(train),
               "lr": a.lr, "batch": a.batch, "minutes": round((time.time() - t0) / 60, 1),
               "stopped_on_budget": stopped},
              open(HERE / "results_finetune.json", "w"), indent=1)
    print(f"\nsaved to {a.out}; re-run pleias_gate.py against it to measure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
