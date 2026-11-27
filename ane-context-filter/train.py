"""Distil Jev's chunk scores into an Ettin cross-encoder small enough for the Neural Engine.

Input per (delegation, chunk) pair, in one sequence of at most MAX_LEN tokens:
    [CLS] task (head and tail, at most TASK_TOKENS) [SEP] <kind tool | age> chunk view [SEP]
`age` is how many chunks before the delegation this one sits (bucketed): recency is
the strongest cheap signal in the baselines. The chunk view is the shipped filter's
1x view (1,500 chars, or 300 of tool input and 900 of result). Target: Jev's score,
as a soft label for BCE. Readout: the [CLS] vector through a small MLP head.

Usage: EVAL_DATA=<private dir> python3 train.py [--init jhu-clsp/ettin-encoder-17m]
         [--epochs 2] [--lr 8e-5] [--bs 16] [--name student17]
Checkpoints every --ckpt-every steps to <EVAL_DATA>/models/<name>/ and resumes from it.
After each epoch it scores the val split and writes <EVAL_DATA>/scores/<name>_val.jsonl;
after the last, the test split too. evaluate.py --scores reads those files.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("subagent-context-filter")))
import jevfilter  # noqa: E402
from evaluate import context, split_of  # noqa: E402

DATA = Path(os.environ.get("EVAL_DATA") or HERE / "data")
MAX_LEN = 384
TASK_TOKENS = 120
AGE_BUCKETS = (1, 3, 10, 30, 100, 300)


def age_label(age):
    for b in AGE_BUCKETS:
        if age <= b:
            return f"age<={b}"
    return f"age>{AGE_BUCKETS[-1]}"


def chunk_header(c, age):
    tool = f" {c.get('tool', '?')}" if c["kind"] == "tool" else ""
    return f"<{c['kind']}{tool} | {age_label(age)}> "


def chunk_body(c):
    v = jevfilter.view(c, 1.0)
    if c["kind"] == "tool":
        return f"input: {v['input']}\nresult: {v['result']}"
    return v["text"]


class Encoder:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, task, c, age):
        t = self.tok(task, add_special_tokens=False)["input_ids"]
        if len(t) > TASK_TOKENS:
            t = t[: TASK_TOKENS * 2 // 3] + t[-(TASK_TOKENS - TASK_TOKENS * 2 // 3):]
        b = self.tok(chunk_header(c, age) + chunk_body(c), add_special_tokens=False)["input_ids"]
        room = MAX_LEN - len(t) - 3
        b = b[:room]
        return [self.tok.cls_token_id] + t + [self.tok.sep_token_id] + b + [self.tok.sep_token_id]


class Scorer(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.enc = AutoModel.from_pretrained(init, attn_implementation="sdpa")
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1))

    def forward(self, ids, att):
        return self.head(self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]).squeeze(-1)


def collate(items, pad):
    L = max(len(x["ids"]) for x in items)
    ids = torch.full((len(items), L), pad, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, x in enumerate(items):
        ids[i, : len(x["ids"])] = torch.tensor(x["ids"]); att[i, : len(x["ids"])] = 1
    return ids, att


def load_examples(split, enc):
    """One example per (delegation, context chunk) of the split, with Jev's score."""
    labels = {}
    for line in open(DATA / "labels.jsonl"):
        r = json.loads(line)
        if "error" not in r:
            labels[r["key"]] = r["scores"]
    rows = [json.loads(l) for l in open(DATA / "delegations.jsonl")]
    rows = [r for r in rows if f"{r['session']}:{r['deleg_chunk']}" in labels and split_of(r["session"]) == split]
    cache, out = {}, []
    for r in rows:
        if r["session"] not in cache:
            cache[r["session"]] = json.loads((DATA / "chunks" / f"{r['session']}.json").read_text())
        ctx = context(cache[r["session"]], r["deleg_chunk"])
        sc = labels[f"{r['session']}:{r['deleg_chunk']}"]
        n = len(ctx)
        for i, c in enumerate(ctx):
            y = sc.get(str(c["id"]))
            if y is None:
                continue
            out.append({"key": f"{r['session']}:{r['deleg_chunk']}", "cid": c["id"],
                        "ids": enc(r["task"], c, n - i), "y": float(y)})
    return out


def length_batches(data, bs, rng):
    """Shuffle, then sort within pools of 50 batches by length: less padding on CPU."""
    idx = list(range(len(data)))
    rng.shuffle(idx)
    pool = bs * 50
    out = []
    for s in range(0, len(idx), pool):
        p = sorted(idx[s: s + pool], key=lambda i: len(data[i]["ids"]))
        out += [p[j: j + bs] for j in range(0, len(p), bs)]
    rng.shuffle(out)
    return out


@torch.no_grad()
def score_split(model, data, pad, path, bs=64):
    model.eval()
    by = {}
    order = sorted(range(len(data)), key=lambda i: len(data[i]["ids"]))
    for s in range(0, len(order), bs):
        items = [data[i] for i in order[s: s + bs]]
        p = torch.sigmoid(model(*collate(items, pad))).tolist()
        for x, v in zip(items, p):
            by.setdefault(x["key"], {})[str(x["cid"])] = round(v, 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, v in by.items():
            f.write(json.dumps({"key": k, "scores": v}) + "\n")
    # pointwise agreement with Jev, for the log only
    ys = [x["y"] for x in data]
    ps = [by[x["key"]][str(x["cid"])] for x in data]
    mae = sum(abs(a - b) for a, b in zip(ys, ps)) / max(1, len(ys))
    model.train()
    return round(mae, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="jhu-clsp/ettin-encoder-17m")
    ap.add_argument("--name", default="student17")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.init)
    enc = Encoder(tok)
    t0 = time.time()
    train, val, test = (load_examples(s, enc) for s in ("train", "val", "test"))
    print(f"examples train={len(train)} val={len(val)} test={len(test)} "
          f"({time.time() - t0:.0f}s to tokenize)", flush=True)
    model = Scorer(a.init)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=0.01)
    plan = [length_batches(train, a.bs, random.Random(a.seed + e)) for e in range(a.epochs)]
    total = sum(map(len, plan))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    mdir = DATA / "models" / a.name
    mdir.mkdir(parents=True, exist_ok=True)
    step = 0
    ck = mdir / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed at step {step}/{total}", flush=True)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    done = 0
    for e, batches in enumerate(plan):
        for b in batches:
            done += 1
            if done <= step:
                continue
            items = [train[i] for i in b]
            ids, att = collate(items, tok.pad_token_id)
            loss = lossf(model(ids, att), torch.tensor([x["y"] for x in items]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step = done
            if step % 50 == 0:
                print(f"epoch {e} step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step}, ck)
            if a.max_steps and step >= a.max_steps:
                print("max steps reached", flush=True)
                return
        mae = score_split(model, val, tok.pad_token_id, DATA / "scores" / f"{a.name}_val.jsonl")
        print(f"epoch {e} done: val MAE vs Jev {mae} {time.time() - t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step}, ck)
    torch.save(model.state_dict(), mdir / "final.pt")
    mae = score_split(model, test, tok.pad_token_id, DATA / "scores" / f"{a.name}_test.jsonl")
    print(f"done: test MAE vs Jev {mae} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
