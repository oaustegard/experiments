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

from encoder import BekkoEncoder, SignBits, condition
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


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_cos(model, tok, texts, emb, n, cond, sb, enc, bs=64):
    """Greedy zero-step decode of n dev items; verifier cosine + exact match. The
    underfit diagnostic: if this matches the same number on the training set,
    the base is not memorising, it is short of capacity or data."""
    model.eval()
    hyps = []
    with torch.no_grad():
        for s in range(0, n, bs):
            ids = model.generate(emb[s:min(s + bs, n)], num_beams=1)
            hyps += tok.batch_decode(ids, skip_special_tokens=True)
    he = condition(cond, enc.encode(hyps, batch_size=64), sb)
    tgt = emb[:n].detach().cpu().numpy()
    cos = float((he * tgt).sum(1).mean())
    exact = float(np.mean([" ".join(a.lower().split()) == " ".join(b.lower().split()) for a, b in zip(texts[:n], hyps)]))
    return cos, exact


def run_epoch(model, tok, texts, emb, hyps, hyp_emb, bs, opt, sched, train, log, tag, *, dev=None, bf16=False,
              seed=0, start_step=0, save_cb=None, save_every=0):
    model.train(train)
    order = np.random.default_rng(seed).permutation(len(texts)) if train else np.arange(len(texts))
    dev = dev or torch.device("cpu")
    tot, n, t0 = 0.0, 0, time.time()
    for step, s in enumerate(range(0, len(texts), bs)):
        if step < start_step:
            continue  # resuming mid-epoch: same permutation, skip what was done
        idx = order[s:s + bs]
        labels, _, _ = encode_labels(tok, [texts[i] for i in idx])
        kw = {}
        if model.mode == "correct":
            _, hid, hm = encode_labels(tok, [hyps[i] for i in idx])
            kw = {"e_hyp": hyp_emb[idx].to(dev), "hyp_ids": hid.to(dev), "hyp_mask": hm.to(dev)}
        with torch.set_grad_enabled(train), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(emb[idx].to(dev), labels.to(dev), **kw)
        if train:
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        tot += out.loss.item() * len(idx); n += len(idx)
        if train and step % 50 == 0:
            print(f"[{tag}] step {step} loss {out.loss.item():.3f} "
                  f"{(time.time()-t0)/(step-start_step+1):.2f}s/step", flush=True)
        if train and save_every and save_cb and step > start_step and step % save_every == 0:
            save_cb(step)
    return tot / max(n, 1)


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
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (mps / cuda)")
    ap.add_argument("--eval-cos", type=int, default=0,
                    help="after each epoch, greedy-decode this many dev items and report verifier cosine + exact")
    ap.add_argument("--save-every", type=int, default=0, help="also checkpoint every N steps (long epochs)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    dev = pick_device(a.device)
    print(f"device {dev}, bf16 {a.bf16}", flush=True)
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

    model = Inverter(a.mode, k=a.k).to(dev)
    enc = BekkoEncoder() if a.eval_cos else None
    if a.init:
        sd = torch.load(a.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm start from {a.init}: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.lr_proj), weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr_t) / a.bs)
    warm = max(1, min(100, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / steps))

    tag = f"{a.mode}_{a.cond}"
    log = {"args": vars(a), "epochs": []}
    best = float("inf")
    start, start_step = 0, 0
    last = CKPT / f"{tag}.last.pt"
    if last.exists():  # a killed run resumes at the last checkpoint (epoch boundary, or --save-every step)
        st = torch.load(last, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        log, best = st["log"], st["best"]
        if st.get("step") is not None:
            start, start_step = st["epoch"], st["step"] + 1
            print(f"[{tag}] resumed inside epoch {start} at step {start_step} (best dev {best:.3f})", flush=True)
        else:
            start = st["epoch"] + 1
            print(f"[{tag}] resumed after epoch {st['epoch']} (best dev {best:.3f})", flush=True)

    def save_last(ep, step=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "log": log, "best": best, "epoch": ep, "step": step}, last)

    for ep in range(start, a.epochs):
        t = time.time()
        trl = run_epoch(model, tok, tr_t, tr_e, tr_h, tr_he, a.bs, opt, sched, True, log, f"{tag} ep{ep}",
                        dev=dev, bf16=a.bf16, seed=a.seed + ep, start_step=start_step if ep == start else 0,
                        save_cb=lambda step: save_last(ep, step), save_every=a.save_every)
        dvl = run_epoch(model, tok, dv_t, dv_e, dv_h, dv_he, a.bs, opt, sched, False, log, tag, dev=dev, bf16=a.bf16)
        rec = {"epoch": ep, "train_loss": trl, "dev_loss": dvl, "secs": time.time() - t}
        if a.eval_cos and a.mode == "zero":
            n = min(a.eval_cos, len(dv_t))
            rec["dev_cos"], rec["dev_exact"] = eval_cos(model, tok, dv_t, dv_e.to(dev), n, a.cond, sb, enc)
            rec["train_cos"], rec["train_exact"] = eval_cos(model, tok, tr_t, tr_e.to(dev), n, a.cond, sb, enc)
            print(f"[{tag}] epoch {ep}: greedy cosine dev {rec['dev_cos']:.3f} train {rec['train_cos']:.3f} "
                  f"exact dev {rec['dev_exact']:.3f} train {rec['train_exact']:.3f} (n={n})", flush=True)
        log["epochs"].append(rec)
        print(f"[{tag}] epoch {ep}: train {trl:.3f} dev {dvl:.3f} ({time.time()-t:.0f}s)", flush=True)
        if dvl < best:
            best = dvl
            torch.save(model.state_dict(), CKPT / f"{tag}.pt")
        save_last(ep)
        (LOGS / f"train_{tag}.json").write_text(json.dumps(log, indent=1))
    print(f"[{tag}] done, best dev {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
