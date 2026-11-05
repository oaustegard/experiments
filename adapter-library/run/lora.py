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
from shared import (FAM_IDS, FAM_WORKFLOW, GPU_SEEDS, ROOT, SIZES, WORKFLOWS, examples, family_map,  # noqa: E402
                    load_cases, revealed, router_example, router_fold, score_examples)
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


@torch.no_grad()
def fake_quant(model):
    """Round the frozen base to what pack.py ships (v0.1.1): int8 per output column (codes in +-63) for
    every Wqkv/Wo/Wi/Wo_mlp, and the token table to 4 bits in groups of 32 with an fp16 scale."""
    for _, mod in model.enc.named_modules():
        for cn, ch in mod.named_children():
            lin = ch.base if isinstance(ch, LoRALinear) else ch
            if cn in TARGETS and isinstance(lin, nn.Linear):
                W = lin.weight
                sc = W.abs().amax(1, keepdim=True) / 63 + 1e-12
                lin.weight.copy_(torch.clamp(torch.round(W / sc), -63, 63) * sc)
    T = model.enc.embeddings.tok_embeddings.weight
    V, H = T.shape
    g = T.view(V, H // 32, 32)
    sc = g.abs().amax(2, keepdim=True) / 7 + 1e-12
    q = torch.clamp(torch.round(g / sc), -8, 7)
    T.copy_((q * sc.half().float()).view(V, H))
    return model


def new_model(kind, qbase=False):
    m = Decider(stock_path())
    if kind != "full":
        add_lora(m)
    if qbase:
        fake_quant(m)
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
    for w in WORKFLOWS:
        for n in CV_SIZES:
            for s in GPU_SEEDS:
                for kind in ("scratch", "warm"):
                    A.append(("cv", f"CV-{kind}-{w}-n{n}-s{s}", w, s, n))
    # round 4b: router spikes (one per held-out family) and one sub-spike per family
    # system v0.1.1: whole-workflow spikes trained on the quantized base
    for w in WORKFLOWS:
        A.append(("spike", f"Q-lora-{w}-s0", w, 0, None))
    # v0.1.1: quantization-aware fine-tuning, starting from round 1's fp32-base spike (QW = warm)
    for w in WORKFLOWS:
        A.append(("spike", f"QW-lora-{w}-s0", w, 0, None))
    for i in range(len(FAM_IDS)):
        A.append(("router", f"R-router-F{i:02d}", None, 0, i))
    for fid in FAM_IDS:
        A.append(("sub", f"R-sub-{fid}", FAM_WORKFLOW[fid], 0, None))
    # round 4a: the cross-fit with 8 folds
    for w in WORKFLOWS:
        for n in CV_SIZES:
            for s in GPU_SEEDS:
                for kind in ("scratch", "warm"):
                    A.append(("cv8", f"CV8-{kind}-{w}-n{n}-s{s}", w, s, n))
    return A


CV_SIZES, FOLDS = (16, 64), 4


def run_cv(name, w, seed, n, kind, tok, ms, t0, FOLDS=FOLDS):
    """Out-of-fold spike predictions on W's revealed cases (round 3: 4 folds; round 4a: 8)."""
    rev = revealed(w, seed, n)
    fold = {c["id"]: i % FOLDS for i, c in enumerate(rev)}
    ex = prep(tok, rev)
    oof = [None] * len(ex)
    lib = None
    if kind == "warm":
        lib = merged_adapter([torch.load(OUT / "adapters" / f"S-lora-{x}-s0.pt") for x in WORKFLOWS if x != w])
    for f in range(FOLDS):
        torch.manual_seed(seed * 100 + f)
        model = new_model(kind)
        if lib is not None:
            load_adapter(model, lib)
        tr = [e for e in ex if fold[e["case"]] != f]
        idx = [i for i, e in enumerate(ex) if fold[e["case"]] == f]
        train(model, "lora", tr, growth_epochs(n - n // FOLDS), seed * 100 + f, max_steps=ms)
        for i, z in zip(idx, predict(model, [ex[i] for i in idx], PAD)):
            oof[i] = probs_at([z], 1.0)[0]
    summ, _ = score_examples(ex, oof)
    (OUT / "probs" / f"{name}.json").write_text(json.dumps([[round(x, 5) for x in p] for p in oof]))
    (RES / f"{name}.json").write_text(json.dumps({"name": name, "workflow": w, "seed": seed, "n": n, "kind": kind,
                                                  "folds": FOLDS, "oof": summ, "sec": time.time() - t0}, indent=1))
    print(f"[{name}] oof acc {summ['accuracy']:.4f} {time.time() - t0:.0f}s", flush=True)


def run_router(name, i, tok, ms, t0, smoke=False):
    """Round 4b: a router spike with family F held out and family G trained as `none`."""
    F, G = router_fold(i)
    fm = family_map()
    lib = [f for f in FAM_IDS if f not in (F, G)]
    with_f = [f for f in FAM_IDS if f != G]
    tr_cases = [c for w in WORKFLOWS for c in load_cases(w, "train") if fm[c["id"]] in lib + [G]]
    tr_cases.sort(key=lambda c: c["id"])
    random.Random(f"router-{i}").shuffle(tr_cases)
    if smoke:
        tr_cases = tr_cases[:40]
    ex = [router_example(c, lib, fm[c["id"]] if fm[c["id"]] != G else "none") for c in tr_cases]
    ex = prep_raw(tok, ex)
    nval = max(20, len(ex) // 10)
    va, tr = ex[:nval], ex[nval:]
    torch.manual_seed(i)
    model = new_model("lora")
    info = train(model, "lora", tr, 10, i, va=va, max_steps=ms)
    te_cases = [c for w in WORKFLOWS for c in load_cases(w, "test")][: 8 if smoke else None]
    out = {"name": name, "F": F, "G": G, "library": lib, "with_F": with_f, **info, "cases": [c["id"] for c in te_cases]}
    for tag, opts in (("A", lib), ("B", with_f)):
        te = prep_raw(tok, [router_example(c, opts, None) for c in te_cases])
        out[f"options_{tag}"] = opts + ["none"]
        out[f"probs_{tag}"] = [[round(x, 5) for x in probs_at([z], 1.0)[0]] for z in predict(model, te, PAD)]
    lab = [fm[c] for c in out["cases"]]
    okA = [out["options_A"][int(np.argmax(p))] == l for p, l in zip(out["probs_A"], lab) if l in lib]
    out["routing_acc_A"] = float(np.mean(okA)) if okA else None
    out["sec"] = time.time() - t0
    (OUT / "probs" / f"{name}.json").write_text(json.dumps(out))
    (RES / f"{name}.json").write_text(json.dumps({k: v for k, v in out.items() if not k.startswith("probs") and k != "cases"},
                                                 indent=1))
    print(f"[{name}] F={F} G={G} routing acc (library, A) {out['routing_acc_A']} {out['sec']:.0f}s", flush=True)


def run_sub(name, fid, w, tok, ms, t0, test_of):
    """Round 4b: a sub-spike trained on one family's train cases, scored on its workflow's test set."""
    fm = family_map()
    cases = [c for c in load_cases(w, "train") if fm[c["id"]] == fid]
    tr = prep(tok, cases)
    torch.manual_seed(0)
    model = new_model("scratch")
    info = train(model, "lora", tr, growth_epochs(len(cases)), 0, max_steps=ms)
    te = test_of[w]
    probs = [probs_at([z], 1.0)[0] for z in predict(model, te, PAD)]
    own = [i for i, e in enumerate(te) if fm[e["case"]] == fid]
    summ = score_examples(te, probs)[0]
    summ_own = score_examples([te[i] for i in own], [probs[i] for i in own])[0] if own else None
    (OUT / "probs" / f"{name}.json").write_text(json.dumps([[round(x, 5) for x in p] for p in probs]))
    (RES / f"{name}.json").write_text(json.dumps({"name": name, "family": fid, "workflow": w, "n_train_cases": len(cases),
                                                  "test_all": summ, "test_own_family": summ_own, **info,
                                                  "sec": time.time() - t0}, indent=1))
    print(f"[{name}] {len(cases)} cases, own-family acc {summ_own and summ_own['accuracy']} {time.time() - t0:.0f}s", flush=True)


def prep_raw(tok, exs):
    out = []
    for e in exs:
        ids, pos = tok.build(e)
        out.append({**e, "ids": ids, "pos": pos})
    return out


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
    prefixes = tuple(a.only.split(",")) if a.only else ("",)
    todo = [x for x in arms() if x[1].startswith(prefixes)]
    if a.smoke:
        w0 = WORKFLOWS[0]
        todo = [x for x in todo if x[3] == 0 and ((x[0] == "spike" and "-lora-" in x[1]) or
                                                  (x[2] == w0 and x[4] in (0, 16)))]
        if a.only:
            todo = [x for p in prefixes for x in [y for y in arms() if y[1].startswith(p) and y[3] == 0
                                                  and y[4] in (16, 0, None)][:2]]
    test_of = {w: prep(tok, load_cases(w, "test")[: 4 if a.smoke else None]) for w in WORKFLOWS}
    for mode, name, w, seed, n in todo:
        if (RES / f"{name}.json").exists():
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        ms = 2 if a.smoke else 0
        if mode == "cv":
            run_cv(name, w, seed, n, name.split("-")[1], tok, ms, t0)
            continue
        if mode == "cv8":
            run_cv(name, w, seed, n, name.split("-")[1], tok, ms, t0, FOLDS=8)
            continue
        if mode == "router":
            run_router(name, n, tok, ms, t0, smoke=a.smoke)
            continue
        if mode == "sub":
            run_sub(name, name[len("R-sub-"):], w, tok, ms, t0, test_of)
            continue
        if mode == "spike":
            kind = "lora" if "-lora-" in name else "full"
            cases = sorted(load_cases(w, "train"), key=lambda c: c["id"])
            random.Random(f"{w}-{seed}").shuffle(cases)
            nval = 30
            va, tr = prep(tok, cases[:nval][: 4 if a.smoke else None]), prep(tok, cases[nval:])
            model = new_model(kind, qbase=name.startswith(("Q-", "QW-")))
            if name.startswith("QW-"):
                load_adapter(model, torch.load(OUT / "adapters" / f"S-lora-{w}-s{seed}.pt"))
            info = train(model, kind, tr, 5 if name.startswith("QW-") else 10, seed, va=va, max_steps=ms)
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
