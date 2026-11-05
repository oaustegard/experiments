"""Fine-tune an Ettin encoder as a typed-decision cross-encoder.

    python3 -I run/train_enc.py NAME --init {e17c,stock} [--train W ...] [--test W ...]
        [--epochs 5] [--lr 8e-5] [--max-steps N]

One sequence per question:
  [CLS] type: instructions [SEP] [MASK] opt: desc [MASK] opt: desc ... [SEP] state [SEP]
A 2-layer MLP scores the final hidden state at each [MASK]; softmax over options;
soft cross-entropy against the gold distribution. Best epoch by validation KL
(10% of train cases), then one temperature per question type fitted on validation.
Writes results/<NAME>.json, results/<NAME>_preds.json and out/ckpt/<NAME>/.
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
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INITS = {
    "e17c": ROOT / "out/base/st-ettin17m-tuned-e17-cap-c-epoch1/epoch1",
    "stock": "jhu-clsp/ettin-encoder-17m",
    "monad": "PleIAs/Monad",
}
CAUSAL = {"monad"}
MAX_LEN = 1024


class Enc:
    def __init__(self, tok_path):
        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.cls, self.sep, self.mask = (self.tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[MASK]"))
        assert None not in (self.cls, self.sep, self.mask)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def build(self, e):
        ids = [self.cls] + self.ids(e["question"]) + [self.sep]
        pos = []
        for name, desc in e["options"]:
            pos.append(len(ids))
            ids += [self.mask] + self.ids(f" {name}: {desc}")
        ids.append(self.sep)
        budget = MAX_LEN - len(ids) - 1
        ids += self.ids(e["state"])[:budget] + [self.sep]
        return ids, pos


class DecTok(Enc):
    """Causal layout (Monad): state first, then the question, then one line per option; each
    option is scored at its line's last token, which has read the state, the question and the
    options before it.  <bos> state \n\n type: instructions \n opt: desc \n opt: desc \n ..."""

    def __init__(self, tok_path):
        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.bos = self.tok.token_to_id("<|begin_of_text|>")
        assert self.bos is not None

    def build(self, e):
        tail = self.ids("\n\n" + e["question"] + "\n")
        pos = []
        for name, desc in e["options"]:
            tail += self.ids(f"{name}: {desc}\n")
            pos.append(len(tail) - 1)
        budget = MAX_LEN - len(tail) - 1
        head = [self.bos] + self.ids(e["state"])[:budget]
        return head + tail, [len(head) + p for p in pos]


class Decider(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.enc = AutoModel.from_pretrained(str(init), attn_implementation="sdpa")
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1))

    def forward(self, ids, att, pos_b, pos_t):
        hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(hs[pos_b, pos_t]).squeeze(-1)  # one logit per option, flat


def collate(batch, pad_id):
    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    pb, pt, seg, gold = [], [], [], []
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        att[i, : len(b["ids"])] = 1
        pb += [i] * len(b["pos"])
        pt += b["pos"]
        seg.append(len(b["pos"]))
        gold += b["gold"]
    return ids, att, torch.tensor(pb), torch.tensor(pt), seg, torch.tensor(gold)


def batches(data, bs, shuffle, rng):
    idx = sorted(range(len(data)), key=lambda i: len(data[i]["ids"]))
    chunks = [idx[i: i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def seg_softmax(logits, seg, T=None):
    out, i = [], 0
    for k in seg:
        z = logits[i: i + k]
        out.append(torch.log_softmax(z / (1.0 if T is None else T), -1))
        i += k
    return out


@torch.no_grad()
def predict(model, data, pad_id, bs=32):
    model.eval()
    logits = [None] * len(data)
    for ch in batches(data, bs, False, None):
        ids, att, pb, pt, seg, _ = collate([data[i] for i in ch], pad_id)
        z = model(ids.to(DEV), att.to(DEV), pb.to(DEV), pt.to(DEV)).float().cpu()
        j = 0
        for i, k in zip(ch, seg):
            logits[i] = z[j: j + k].numpy().astype(np.float64)
            j += k
    return logits


def probs_at(logits, T):
    out = []
    for z in logits:
        z = z / T
        z = np.exp(z - z.max())
        out.append((z / z.sum()).tolist())
    return out


def kl_mean(data, probs):
    tot = 0.0
    for d, p in zip(data, probs):
        tot += sum(g * math.log(g / max(q, 1e-9)) for g, q in zip(d["gold"], p) if g > 0)
    return tot / len(data)


def fit_temps(data, logits):
    temps = {}
    grid = np.exp(np.linspace(math.log(0.3), math.log(4.0), 60))
    for t in sorted({d["type"] for d in data}):
        sel = [i for i, d in enumerate(data) if d["type"] == t]
        best = min(grid, key=lambda T: kl_mean([data[i] for i in sel], probs_at([logits[i] for i in sel], T)))
        temps[t] = float(best)
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--init", choices=list(INITS), default="stock")
    ap.add_argument("--train", nargs="+", default=WORKFLOWS)
    ap.add_argument("--test", nargs="+", default=WORKFLOWS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    a = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    t0 = time.time()

    init = INITS[a.init]
    if not Path(str(init)).exists():  # a Hub id (stock Ettin, Monad)
        from huggingface_hub import snapshot_download
        init = snapshot_download(init, allow_patterns=["*.json", "*.bin", "*.safetensors"])
    tok = (DecTok if a.init in CAUSAL else Enc)(Path(init) / "tokenizer.json")
    pad_id = tok.tok.token_to_id("[PAD]")
    assert pad_id is not None

    train_cases = [c for w in a.train for c in load_cases(w, "train")]
    rng.shuffle(train_cases)
    nval = max(20, len(train_cases) // 10)
    val_cases, tr_cases = train_cases[:nval], train_cases[nval:]
    test_cases = [c for w in a.test for c in load_cases(w, "test")]

    def prep(cases):
        out = []
        for e in examples(cases):
            ids, pos = tok.build(e)
            out.append({**e, "ids": ids, "pos": pos})
        return out

    tr, va, te = prep(tr_cases), prep(val_cases), prep(test_cases)
    print(f"[{a.name}] train {len(tr)} val {len(va)} test {len(te)} examples; "
          f"max len {max(len(d['ids']) for d in tr + va + te)}", flush=True)

    model = Decider(init).to(DEV)
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": a.lr},
        {"params": head_params, "lr": a.lr * 10},
    ], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr) / a.bs)
    total = steps_per_epoch * a.epochs
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))

    ckdir = ROOT / "out" / "ckpt" / a.name
    ckdir.mkdir(parents=True, exist_ok=True)
    log, best, step = [], (float("inf"), -1), 0
    for ep in range(a.epochs):
        model.train()
        te0, losses = time.time(), []
        for ch in batches(tr, a.bs, True, rng):
            ids, att, pb, pt, seg, gold = collate([tr[i] for i in ch], pad_id)
            z = model(ids.to(DEV), att.to(DEV), pb.to(DEV), pt.to(DEV))
            gold = gold.to(DEV)
            lp = torch.cat(seg_softmax(z, seg))
            loss = -(gold * lp).sum() / len(seg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"[{a.name}] ep {ep} step {step}/{total} loss {np.mean(losses[-50:]):.4f} "
                      f"{(time.time() - te0) / len(losses):.2f}s/step", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        vl = predict(model, va, pad_id)
        vkl = kl_mean(va, probs_at(vl, 1.0))
        vsum, _ = score_examples(va, probs_at(vl, 1.0))
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_kl": vkl,
                    "val_acc": vsum["accuracy"], "sec": time.time() - te0})
        print(f"[{a.name}] epoch {ep}: {log[-1]}", flush=True)
        if vkl < best[0]:
            best = (vkl, ep)
            torch.save(model.state_dict(), ckdir / "best.pt")
        if a.max_steps and step >= a.max_steps:
            break

    model.load_state_dict(torch.load(ckdir / "best.pt", map_location=DEV))
    vl = predict(model, va, pad_id)
    temps = fit_temps(va, vl)
    tl = predict(model, te, pad_id)
    probs = [probs_at([z], temps[d["type"]])[0] for z, d in zip(tl, te)]
    raw = probs_at(tl, 1.0)
    summ, rows = score_examples(te, probs)
    summ_raw, _ = score_examples(te, raw)
    by = {}
    for key in ("workflow", "type"):
        for v in sorted({d[key] for d in te}):
            sel = [i for i, d in enumerate(te) if d[key] == v]
            by[f"{key}={v}"] = score_examples([te[i] for i in sel], [probs[i] for i in sel])[0]
    for w in sorted({d["workflow"] for d in te}):
        for q in sorted({d["qname"] for d in te if d["workflow"] == w}):
            sel = [i for i, d in enumerate(te) if d["workflow"] == w and d["qname"] == q]
            by[f"q={w}/{q}"] = score_examples([te[i] for i in sel], [probs[i] for i in sel])[0]
    res = {"name": a.name, "args": vars(a), "best_epoch": best[1], "temps": temps,
           "test": summ, "test_raw_T1": summ_raw, "by": by, "log": log,
           "wall_sec": time.time() - t0, "device": str(DEV), "n_params": sum(p.numel() for p in model.parameters())}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    (ROOT / "results" / f"{a.name}_preds.json").write_text(json.dumps(
        [{"case": d["case"], "q": d["qname"], "p": [round(x, 5) for x in p], "ok": r[0]}
         for d, p, r in zip(te, probs, rows)]))
    (ckdir / "meta.json").write_text(json.dumps({"init": str(init), "temps": temps}))
    print(f"[{a.name}] TEST {json.dumps(summ)} (raw T=1: acc {summ_raw['accuracy']:.4f}) "
          f"wall {res['wall_sec'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
