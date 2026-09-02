"""Train the zero-step inverter or the corrector for one condition.

  python3 train.py --mode zero    --cond float --epochs 3
  python3 train.py --mode correct --cond float --epochs 1 --hyps data/hyps_float_train.json

Under `--cond bin1` the model sees the dequantized 1-bit code of the target,
and (for the corrector) of the hypothesis too, so the verifier and the model
see the same thing the index would. Checkpoints land in ckpt/ (gitignored);
per-epoch dev loss in logs/train_<mode>_<cond>.json (committed).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from encoder import SignBits, condition
from model import Inverter, encode_labels, tokenizer

HERE = Path(__file__).resolve().parent
DATA, CKPT, LOGS = HERE / "data", HERE / "ckpt", HERE / "logs"


def load_split(name, cond, sb):
    texts = json.loads((DATA / "splits.json").read_text())[name]
    emb = condition(cond, np.load(DATA / f"emb_{name}.npy"), sb)
    return texts, torch.from_numpy(emb)


def load_hyps(path, cond, sb):
    h = json.loads(Path(path).read_text())
    emb = condition(cond, np.array(h["emb"], dtype=np.float32), sb)
    return h["hyp"], torch.from_numpy(emb)


def run_epoch(model, tok, texts, emb, hyps, hyp_emb, bs, opt, sched, train, log, tag):
    model.train(train)
    order = np.random.permutation(len(texts)) if train else np.arange(len(texts))
    tot, n, t0 = 0.0, 0, time.time()
    for step, s in enumerate(range(0, len(texts), bs)):
        idx = order[s:s + bs]
        labels, _, _ = encode_labels(tok, [texts[i] for i in idx])
        kw = {}
        if model.mode == "correct":
            _, hid, hm = encode_labels(tok, [hyps[i] for i in idx])
            kw = {"e_hyp": hyp_emb[idx], "hyp_ids": hid, "hyp_mask": hm}
        with torch.set_grad_enabled(train):
            out = model(emb[idx], labels, **kw)
        if train:
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        tot += out.loss.item() * len(idx); n += len(idx)
        if train and step % 50 == 0:
            print(f"[{tag}] step {step} loss {out.loss.item():.3f} "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero", "correct"], required=True)
    ap.add_argument("--cond", choices=["float", "bin1"], required=True)
    ap.add_argument("--hyps", help="hypotheses json for the corrector (train split)")
    ap.add_argument("--dev-hyps", help="hypotheses json for the corrector (dev split)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=0, help="cap (0 = all)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="checkpoint to warm-start from (zero -> correct)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    CKPT.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)

    sb = SignBits.load(DATA / "signbits_mu.npy") if a.cond == "bin1" else None
    tok = tokenizer()
    tr_t, tr_e = load_split("train", a.cond, sb)
    dv_t, dv_e = load_split("dev", a.cond, sb)
    if a.n_train:
        tr_t, tr_e = tr_t[: a.n_train], tr_e[: a.n_train]
    tr_h = dv_h = tr_he = dv_he = None
    if a.mode == "correct":
        tr_h, tr_he = load_hyps(a.hyps, a.cond, sb)
        dv_h, dv_he = load_hyps(a.dev_hyps, a.cond, sb)
        assert len(tr_h) >= len(tr_t) and len(dv_h) == len(dv_t)

    model = Inverter(a.mode, k=a.k)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm start from {a.init}: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.lr_proj), weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr_t) / a.bs)
    warm = min(100, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / steps))

    tag = f"{a.mode}_{a.cond}"
    log = {"args": vars(a), "epochs": []}
    best = float("inf")
    for ep in range(a.epochs):
        t = time.time()
        trl = run_epoch(model, tok, tr_t, tr_e, tr_h, tr_he, a.bs, opt, sched, True, log, f"{tag} ep{ep}")
        dvl = run_epoch(model, tok, dv_t, dv_e, dv_h, dv_he, a.bs, opt, sched, False, log, tag)
        log["epochs"].append({"epoch": ep, "train_loss": trl, "dev_loss": dvl, "secs": time.time() - t})
        print(f"[{tag}] epoch {ep}: train {trl:.3f} dev {dvl:.3f} ({time.time()-t:.0f}s)", flush=True)
        if dvl < best:
            best = dvl
            torch.save(model.state_dict(), CKPT / f"{tag}.pt")
        (LOGS / f"train_{tag}.json").write_text(json.dumps(log, indent=1))
    print(f"[{tag}] done, best dev {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
