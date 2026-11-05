"""GPU arms: LoRA spikes, full fine-tunes, n = 0 fallbacks and growth runs, in one process.

    python3 -I run/lora.py [--only PREFIX] [--smoke]

Every arm writes results/<name>.json (summary) and out/probs/<name>.json (per-test-decision
probabilities); an arm whose results file exists is skipped, so the run resumes. LoRA spikes from S are
kept in out/adapters/<name>.pt for the fallback and warm arms.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import GPU_SEEDS, ROOT, SIZES, WORKFLOWS, examples, load_cases, revealed, score_examples  # noqa: E402
from train_enc import (DEV, Decider, Enc, batches, collate, fit_temps, kl_mean, predict,  # noqa: E402
                       probs_at, seg_softmax)

RANK, ALPHA, TARGETS = 8, 16, {"Wqkv", "Wo", "Wi"}
LR_FULL, LR_LORA = 8e-5, 5e-4
RES, OUT = ROOT / "results", ROOT / "out"
_INIT = None


def stock_path():
    global _INIT
    if _INIT is None:
        from huggingface_hub import snapshot_download
        _INIT = snapshot_download("jhu-clsp/ettin-encoder-17m", allow_patterns=["*.json", "*.bin", "*.safetensors"])
    return _INIT


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def delta(self):
        return (self.B @ self.A) * self.scale

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def lora_modules(model):
    return {n: m for n, m in model.enc.named_modules() if isinstance(m, LoRALinear)}


def add_lora(model):
    for p in model.enc.parameters():
        p.requires_grad_(False)
    for _, mod in list(model.enc.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and cn in TARGETS:
                setattr(mod, cn, LoRALinear(ch))
    return model


def adapter_state(model):
    st = {f"{n}.{k}": v.detach().cpu() for n, m in lora_modules(model).items() for k, v in (("A", m.A), ("B", m.B))}
    st.update({f"head.{k}": v.detach().cpu() for k, v in model.head.state_dict().items()})
    return st


def load_adapter(model, st):
    for n, m in lora_modules(model).items():
        m.A.data.copy_(st[f"{n}.A"])
        m.B.data.copy_(st[f"{n}.B"])
    model.head.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("head.")})


def merged_adapter(sts):
    """Mean of the spikes' weight deltas per layer, reduced to rank RANK by SVD; heads averaged."""
    out, scale = {}, ALPHA / RANK
    for k in sts[0]:
        if k.endswith(".A"):
            n = k[:-2]
            d = sum(s[f"{n}.B"] @ s[f"{n}.A"] for s in sts) * scale / len(sts)
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            r = S[:RANK].sqrt() / math.sqrt(scale)
            out[f"{n}.B"] = U[:, :RANK] * r
            out[f"{n}.A"] = r[:, None] * Vh[:RANK]
        elif k.startswith("head."):
            out[k] = sum(s[k] for s in sts) / len(sts)
    return out


def new_model(kind):
    m = Decider(stock_path())
    if kind != "full":
        add_lora(m)
    return m.to(DEV)


def prep(tok, cases):
    out = []
    for e in examples(cases):
        ids, pos = tok.build(e)
        out.append({**e, "ids": ids, "pos": pos})
    return out


def train(model, kind, tr, epochs, seed, bs=16, va=None, max_steps=0):
    rng = random.Random(seed)
    lr = LR_FULL if kind == "full" else LR_LORA
    body = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": lr},
                             {"params": list(model.head.parameters()), "lr": lr * (10 if kind == "full" else 1.6)}],
                            weight_decay=0.01)
    total = math.ceil(len(tr) / bs) * epochs
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    pad = PAD
    best, best_st, log, step = (float("inf"), -1), None, [], 0
    for ep in range(epochs):
        model.train()
        for ch in batches(tr, bs, True, rng):
            ids, att, pb, pt, seg, gold = collate([tr[i] for i in ch], pad)
            z = model(ids.to(DEV), att.to(DEV), pb.to(DEV), pt.to(DEV))
            loss = -(gold.to(DEV) * torch.cat(seg_softmax(z, seg))).sum() / len(seg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if max_steps and step >= max_steps:
                break
        if va is not None:
            vkl = kl_mean(va, probs_at(predict(model, va, pad), 1.0))
            log.append({"epoch": ep, "val_kl": vkl})
            if vkl < best[0]:
                best = (vkl, ep)
                best_st = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if max_steps and step >= max_steps:
            break
    if best_st is not None:
        model.load_state_dict(best_st)
    return {"best_epoch": best[1], "log": log, "steps": step}


def evaluate(name, model, te, temps=None, extra=None):
    lg = predict(model, te, PAD)
    probs = [probs_at([z], (temps or {}).get(d["type"], 1.0))[0] for z, d in zip(lg, te)]
    summ, rows = score_examples(te, probs)
    res = {"name": name, "test": summ, "temps": temps, **(extra or {})}
    (RES / f"{name}.json").write_text(json.dumps(res, indent=1))
    (OUT / "probs" / f"{name}.json").write_text(json.dumps([[round(x, 5) for x in p] for p in probs]))
    print(f"[{name}] acc {summ['accuracy']:.4f} kl {summ['kl_from_gold']:.4f} {extra.get('sec', 0) if extra else 0:.0f}s",
          flush=True)
    return res


def growth_epochs(n, bs=16):
    spe = math.ceil(5 * n / bs)
    return min(40, max(10, math.ceil(150 / spe)))


def arms():
    A = []
    for w in WORKFLOWS:
        for s in (0, 1, 2):
            A += [("spike", f"S-lora-{w}-s{s}", w, s, None), ("spike", f"S-full-{w}-s{s}", w, s, None)]
    for w in WORKFLOWS:
        A.append(("zero", f"G-fallback-{w}", w, 0, 0))
        for n in SIZES[1:]:
            for s in GPU_SEEDS:
                for kind in ("scratch", "warm", "full"):
                    A.append(("grow", f"G-{kind}-{w}-n{n}-s{s}", w, s, n))
    return A


def main():
    global PAD
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true", help="2 steps per arm, 1 workflow, writes under out/smoke")
    a = ap.parse_args()
    global RES, OUT
    if a.smoke:
        RES = OUT = ROOT / "out" / "smoke"
    RES.mkdir(parents=True, exist_ok=True)
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    (OUT / "adapters").mkdir(parents=True, exist_ok=True)
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    PAD = tok.tok.token_to_id("[PAD]")
    todo = [x for x in arms() if x[1].startswith(a.only)]
    if a.smoke:
        w0 = WORKFLOWS[0]
        todo = [x for x in todo if x[3] == 0 and ((x[0] == "spike" and "-lora-" in x[1]) or
                                                  (x[2] == w0 and x[4] in (0, 16)))]
    test_of = {w: prep(tok, load_cases(w, "test")[: 4 if a.smoke else None]) for w in WORKFLOWS}
    for mode, name, w, seed, n in todo:
        if (RES / f"{name}.json").exists():
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        ms = 2 if a.smoke else 0
        if mode == "spike":
            kind = "lora" if "-lora-" in name else "full"
            cases = sorted(load_cases(w, "train"), key=lambda c: c["id"])
            random.Random(f"{w}-{seed}").shuffle(cases)
            nval = 30
            va, tr = prep(tok, cases[:nval][: 4 if a.smoke else None]), prep(tok, cases[nval:])
            model = new_model(kind)
            info = train(model, kind, tr, 10, seed, va=va, max_steps=ms)
            info["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            temps = fit_temps(va, predict(model, va, PAD))
            if kind == "lora":
                torch.save(adapter_state(model), OUT / "adapters" / f"{name}.pt")
            evaluate(name, model, test_of[w], temps, {**info, "kind": kind, "workflow": w, "seed": seed,
                                                      "sec": time.time() - t0})
        elif mode == "zero":
            lib = [x for x in WORKFLOWS if x != w]
            sts = {x: torch.load(OUT / "adapters" / f"S-lora-{x}-s0.pt") for x in lib}
            model = new_model("lora")
            for x, st in sts.items():
                load_adapter(model, st)
                evaluate(f"{name}-spike-{x}", model, test_of[w], None, {"workflow": w, "spike": x, "sec": time.time() - t0})
            load_adapter(model, merged_adapter(list(sts.values())))
            evaluate(f"{name}-merge", model, test_of[w], None, {"workflow": w, "spike": "merge", "sec": time.time() - t0})
            (RES / f"{name}.json").write_text(json.dumps({"name": name, "done": True}))
        else:
            kind = name.split("-")[1]
            model = new_model(kind)
            if kind == "warm":
                load_adapter(model, merged_adapter(
                    [torch.load(OUT / "adapters" / f"S-lora-{x}-s0.pt") for x in WORKFLOWS if x != w]))
            tr = prep(tok, revealed(w, seed, n))
            ep = growth_epochs(n)
            info = train(model, "full" if kind == "full" else "lora", tr, ep, seed, max_steps=ms)
            evaluate(name, model, test_of[w], None, {**info, "kind": kind, "workflow": w, "seed": seed, "n": n,
                                                    "epochs": ep, "sec": time.time() - t0})
    print("ALL_ARMS_DONE", flush=True)


PAD = None
if __name__ == "__main__":
    main()
