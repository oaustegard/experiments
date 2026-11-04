"""Full fine-tune of Monad (A8p weights) on the paper rows plus fresh SYNTH replay rows each epoch.

    python3 train.py --model models/A8p --epochs 10 --lr 1e-4 --save-at 2,4,6,8,10 --out ckpt/lr1e-4
    python3 train.py ... --probe 3          # time three steps and exit

Each row is the chat prompt, then the target `reasoning\n</think>\n\nanswer<|im_end|>` + EOS, with the
loss on the target only (the same masking as monad-bsky/train.py). No BOS: the page and the eval
harness feed the prompt without one. Every epoch sees all paper rows and the next --replay-per-epoch
SYNTH rows from data/replay.jsonl, so replay rows are not repeated until the pool runs out.
--replay-long adds a second pool whose rows are 1,025 to --max-len tokens, --long-per-epoch of them per
epoch, so the model keeps seeing long answers. Checkpoints are saved in bf16, the dtype Monad ships in and the one export.py packs.
Runs on CUDA when present, else CPU.
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def target(r):
    return f"{r['reasoning'].strip()}\n</think>\n\n{r['answer'].strip()}<|im_end|>"


def encode(tok, r, max_len):
    p = tok(prompt(r["q"]), add_special_tokens=False)["input_ids"]
    t = tok(target(r), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids, m = (p + t)[:max_len], ([0] * len(p) + [1] * len(t))[:max_len]
    return ids, m


def collate(batch, pad):
    L = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in batch])
    m = torch.tensor([x + [0] * (L - len(x)) for _, x in batch], dtype=torch.float32)
    att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in batch])
    return ids, m, att


def loss_fn(model, ids, m, att):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none").view(ids[:, 1:].shape)
    mm = m[:, 1:]
    return (tl * mm).sum() / mm.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models/A8p"))
    ap.add_argument("--papers", default=str(HERE / "data/train_papers.jsonl"))
    ap.add_argument("--replay", default=str(HERE / "data/replay.jsonl"))
    ap.add_argument("--replay-per-epoch", type=int, default=300)
    ap.add_argument("--replay-long", default="", help="second replay pool of long rows (fetch_replay.py --min-chars)")
    ap.add_argument("--long-per-epoch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save-at", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--probe", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()   # 64 layers x batch 16 x 1,024 tokens of activations will not fit otherwise
    model.train()
    papers = [json.loads(l) for l in open(a.papers)]
    replay = [json.loads(l) for l in open(a.replay)]
    rng.shuffle(replay)
    P = [encode(tok, r, a.max_len) for r in papers]
    R = [x for x in (encode(tok, r, 4096) for r in replay) if len(x[0]) <= 1024]   # whole rows only, as in rounds 1-2
    L = []
    if a.replay_long:
        longs = [json.loads(l) for l in open(a.replay_long)]
        rng.shuffle(longs)
        L = [x for x in (encode(tok, r, 4096) for r in longs) if 1024 < len(x[0]) <= a.max_len]
    per_epoch = len(P) + a.replay_per_epoch + (a.long_per_epoch if L else 0)
    steps_per_epoch = math.ceil(per_epoch / a.batch_size)
    total = steps_per_epoch * a.epochs
    warmup = max(1, total // 20)
    save_at = {int(x) for x in a.save_at.split(",") if x} | {a.epochs}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    print(json.dumps({"device": dev, "paper_rows": len(P), "replay_pool": len(R), "long_pool": len(L), "per_epoch": per_epoch,
                      "steps": total, "lr": a.lr, "batch": a.batch_size}), flush=True)
    log = []
    step, t0 = 0, time.perf_counter()
    for ep in range(a.epochs):
        k = (ep * a.replay_per_epoch) % len(R)
        rows = P + [(R + R)[k + j] for j in range(a.replay_per_epoch)]
        if L:
            kl = (ep * a.long_per_epoch) % len(L)
            rows += [L[(kl + j) % len(L)] for j in range(a.long_per_epoch)]
        rng.shuffle(rows)
        tot_p = tot_n = 0.0
        for i in range(0, len(rows), a.batch_size):
            ids, m, att = (x.to(dev) for x in collate(rows[i:i + a.batch_size], tok.pad_token_id))
            lr = a.lr * (step + 1) / warmup if step < warmup else a.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            for g in opt.param_groups:
                g["lr"] = lr
            loss = loss_fn(model, ids, m, att)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tot_p += loss.item(); tot_n += 1
            if a.probe and step >= a.probe:
                print(f"probe {step} steps {time.perf_counter() - t0:.1f}s", flush=True)
                return
        log.append({"epoch": ep + 1, "train_loss": tot_p / tot_n, "elapsed_s": time.perf_counter() - t0})
        print(json.dumps(log[-1]), flush=True)
        if ep + 1 in save_at:
            ck = Path(f"{a.out}-e{ep + 1}")
            ck.mkdir(parents=True, exist_ok=True)
            m16 = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
            m16.load_state_dict({k: v.to(torch.bfloat16).cpu() for k, v in model.state_dict().items()})
            m16.config.use_cache = True
            m16.save_pretrained(ck)
            tok.save_pretrained(ck)
            for f in ("chat_template.json", "generation_config.json"):
                p = Path(a.model) / f
                if p.exists():
                    (ck / f).write_bytes(p.read_bytes())
            print("saved", ck, flush=True)
    Path(f"{a.out}.log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
