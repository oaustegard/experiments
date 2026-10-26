#!/usr/bin/env python3
"""Train a 1.5M query encoder into 115M's token space, so 1.5M queries can search a 115M index.

Teacher: 115M, query side only, frozen. Student: every weight of 1.5M, initialised from the
checkpoint. Both see the same token ids (the sizes share the tokenizer and the 32-token query
format), so the loss is per position: 1 - cos(student token, teacher token), averaged over all
32 rows, [MASK] padding included, since MaxSim scores every query row.

Training queries: MS MARCO queries plus Natural Questions, minus any text in the 13 NanoBEIR
query sets. Eval: student queries against 115M documents on NanoBEIR (needs
`encode.py 115M` first, which also gives the teacher's own 0.642 as the ceiling).

Usage: MCB_DEVICE=mps distill.py [--steps N] [--batch B] [--lr LR] [--nq N] [--eval-every N]
Writes .work/distill/student_<step>.safetensors and results/distill.json (appended per eval).
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb import MoEdernColBERT, MASK_ID
from encode import ENC, SETS, DEVICE, ROOT, load_set
from evaluate import maxsim_scores, ndcg10

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("MCB_OUT", HERE.parent / ".work" / "distill"))
RES = Path(os.environ.get("MCB_RES", HERE.parent / "results")) / "distill.json"


def train_queries(n, seed=0):
    from datasets import load_dataset
    held = set()
    for s in SETS:
        held.update(t.strip().lower() for t in load_set(s)[3])
    ms = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")["text"][:]
    nq = load_dataset("sentence-transformers/natural-questions", "pair", split="train")["query"][:]
    qs = list({t for t in list(ms) + list(nq) if t.strip().lower() not in held})
    qs.sort()
    random.Random(seed).shuffle(qs)
    print(f"train queries: {len(qs)} available, using {min(n, len(qs))}; held out {len(held)}", flush=True)
    return qs[:n]


def batch_ids(m, texts):
    seqs = m.ids(texts, "query", prefix=True)
    ids = torch.full((len(seqs), m.ql), MASK_ID, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s)
    return ids.to(m.dev), torch.ones(ids.shape, device=m.dev)


def evaluate(student, sets=SETS):
    """Mean NanoBEIR nDCG@10 of student queries against 115M document tokens."""
    per = {}
    for s in sets:
        meta = json.load(open(ENC / s / "meta.json"))
        z = np.load(ENC / s / "115M_d.npz")
        dt = torch.from_numpy(z["tok"].astype(np.float32)).to(DEVICE)
        qtexts = load_set(s)[3]
        E = student.encode(qtexts, "query", batch_size=256)
        qt = torch.from_numpy(np.concatenate(E)).to(DEVICE)
        qoff = np.arange(len(E) + 1) * student.ql
        per[s] = float(np.mean(ndcg10(maxsim_scores(qt, qoff, dt, z["off"]), meta)))
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nq", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-sets", default=",".join(SETS))
    a = ap.parse_args()
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = [x for x in a.eval_sets.split(",") if x]

    teacher = MoEdernColBERT("115M", device=DEVICE, root=ROOT)
    student = MoEdernColBERT("1.5M", device=DEVICE, root=ROOT)
    params = list(student.w.values()) + [student.dense]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    warm = max(1, a.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

    qs = train_queries(a.nq)
    log = {"args": vars(a), "device": DEVICE, "evals": []}
    if RES.exists():
        log = json.load(open(RES)) | {"args": vars(a), "device": DEVICE}
        log.setdefault("evals", [])

    def do_eval(step, loss):
        if not sets:
            return
        t0 = time.time()
        mean, per = evaluate(student, sets)
        log["evals"].append({"step": step, "loss": loss, "ndcg10": mean, "per_set": per, "sets": len(sets)})
        RES.parent.mkdir(parents=True, exist_ok=True)
        json.dump(log, open(RES, "w"), indent=1)
        print(f"eval step {step}: nDCG@10 {mean:.4f} ({time.time() - t0:.0f}s)", flush=True)

    do_eval(0, None)
    pos, run, t0 = 0, 0.0, time.time()
    for step in range(1, a.steps + 1):
        if pos + a.batch > len(qs):
            pos = 0
        texts = qs[pos:pos + a.batch]; pos += a.batch
        ids, am = batch_ids(student, texts)
        with torch.no_grad():
            tgt = teacher.forward(ids, am)
        out = student.forward_grad(ids, am)
        loss = (1 - (out * tgt).sum(-1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{step * a.batch / el:.0f} q/s, eta {el / step * (a.steps - step) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            save_file({k: v.detach().cpu().contiguous() for k, v in student.w.items()},
                      OUT / f"student_{step}.safetensors")
            save_file({"linear.weight": student.dense.detach().cpu().contiguous()},
                      OUT / f"student_{step}_dense.safetensors")
            do_eval(step, run)


if __name__ == "__main__":
    main()
