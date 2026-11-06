"""PyTorch reference for the WGSL trainer: fixtures for the gradcheck and for a seeded training run.

    python3 -I run/reference.py gradcheck   # fixtures/gradcheck/: one batch, loss, every gradient, one AdamW step
    python3 -I run/reference.py traj        # fixtures/traj/: tokenized data, batch order, initial adapter;
                                            # results/cpu_traj.json: per-step loss and per-epoch metrics

The model is bsky-account-decider's page model rebuilt from its packed weights (thread_lora.base_model),
with its rank-8 LoRA on Wqkv / Wo / Wi and a trainable head. The flat parameter layout (layout()) is the
one web/train.js uses: per layer, A then B for Wqkv, Wo, Wi and Wo_mlp; then the head.
Fixtures hold tokenized Bluesky posts, so fixtures/ is gitignored.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

BSKY = experiment("bsky-account-decider")
sys.path[:0] = [str(BSKY / "run"), str(experiment("browser-decider") / "run")]
import thread_lora as TL  # noqa: E402
from train_enc import batches, collate, seg_softmax  # noqa: E402

FIX = HERE / "fixtures"
TARGETS = [("Wqkv", "attn.Wqkv"), ("Wo", "attn.Wo"), ("Wi", "mlp.Wi"), ("Wo_mlp", "mlp.Wo")]   # add_lora matches child name "Wo", so mlp.Wo is a target too
HEAD = [("w0", "head.0.weight"), ("b0", "head.0.bias"), ("ln_w", "head.2.weight"), ("ln_b", "head.2.bias"),
        ("w1", "head.3.weight"), ("b1", "head.3.bias")]
LR, LR_HEAD, WD, CLIP = 5e-4, 8e-4, 0.01, 1.0
torch.set_num_threads(4)


def tensors(m):
    """(name, parameter) in flat-layout order."""
    mods = dict(m.enc.named_modules())
    out = []
    for i in range(len(m.enc.layers)):
        for short, path in TARGETS:
            mod = mods[f"layers.{i}.{path}"]
            out += [(f"L{i}.{short}.A", mod.A), (f"L{i}.{short}.B", mod.B)]
    hp = dict(m.named_parameters())
    return out + [(f"head.{short}", hp[name]) for short, name in HEAD]


def layout(m):
    off, out = 0, []
    for n, t in tensors(m):
        out.append({"name": n, "shape": list(t.shape), "offset": off})
        off += t.numel()
    return out, off


def flat(m, grad=False):
    return torch.cat([(t.grad if grad else t).detach().reshape(-1) for _, t in tensors(m)]).numpy().astype(np.float32)


def model(seed):
    m, man, _ = TL.base_model()
    torch.manual_seed(seed)
    TL.add_lora(m)
    return m, man


def batch_arrays(data, idx, pad):
    ids, att, pb, pt, seg, gold = collate([data[i] for i in idx], pad)
    B, S = ids.shape
    return {"ids": ids.numpy().astype(np.int32).reshape(-1), "B": B, "S": S,
            "lens": att.sum(1).numpy().astype(np.int32), "pos": (pb * S + pt).numpy().astype(np.int32),
            "seg": np.array(seg, np.int32), "gold": gold.numpy().astype(np.float32)}, (ids, att, pb, pt, seg, gold)


def loss_of(m, tb, T):
    ids, att, pb, pt, seg, gold = tb
    z = m(ids, att, pb, pt) / T
    return -(gold * torch.cat(seg_softmax(z, seg))).sum() / len(seg)


def optimizer(m):
    body = [t for n, t in tensors(m) if not n.startswith("head.")]
    head = [t for n, t in tensors(m) if n.startswith("head.")]
    return torch.optim.AdamW([{"params": body, "lr": LR}, {"params": head, "lr": LR_HEAD}], weight_decay=WD)


def gradcheck():
    m, man = model(0)
    T = man["temps"]["noul"]
    torch.manual_seed(1)
    for n, t in tensors(m):
        if n.endswith(".B"):
            t.data.normal_(0, 0.02)   # nonzero B, so every A gets a gradient
    tok = TL.Enc(TL.M / "tokenizer.json")
    pad = tok.tok.token_to_id("[PAD]")
    D = BSKY / "data" / "thread_lora"
    labels = {}
    for f in sorted((D / "labels").glob("b*.json")):
        labels |= json.loads(f.read_text())["labels"]
    items = [json.loads(l) for l in (D / "items.jsonl").open()]
    items = [it for it in items if it["uri"] in labels]
    data = TL.recs(items[:1500], labels, tok)
    by_len = sorted(range(len(data)), key=lambda i: len(data[i]["ids"]))
    want = [90, 130, 150, 200, 300, 420, 600, 900]   # crosses the local window (129) and pads every row
    idx = [min(by_len, key=lambda i: abs(len(data[i]["ids"]) - w)) for w in want]
    arr, tb = batch_arrays(data, idx, pad)
    p0 = flat(m)
    loss = loss_of(m, tb, T)
    loss.backward()
    g = flat(m, grad=True)
    norm = float(torch.nn.utils.clip_grad_norm_([t for _, t in tensors(m)], CLIP))
    opt = optimizer(m)
    opt.step()
    p1 = flat(m)
    out = FIX / "gradcheck"
    out.mkdir(parents=True, exist_ok=True)
    lay, n = layout(m)
    for k in ("ids", "lens", "pos", "seg", "gold"):
        arr[k].tofile(out / f"{k}.bin")
    p0.tofile(out / "params0.bin")
    g.tofile(out / "grads.bin")
    p1.tofile(out / "params1.bin")
    meta = {"B": arr["B"], "S": arr["S"], "nP": len(arr["pos"]), "nSeg": len(arr["seg"]), "T": T, "loss": float(loss),
            "grad_norm": norm, "layout": lay, "nParams": n, "lens": arr["lens"].tolist(),
            "adam": {"lr": LR, "lr_head": LR_HEAD, "wd": WD, "clip": CLIP, "step": 1}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print({"loss": float(loss), "grad_norm": norm, "nParams": n, "B": arr["B"], "S": arr["S"], "lens": arr["lens"].tolist()})


def splits(tok):
    """thread_lora.train's split: by account, 15% test, 7% val, the rest train; plus the reply_thread set."""
    D = BSKY / "data" / "thread_lora"
    labels = {}
    for f in sorted((D / "labels").glob("b*.json")):
        labels |= json.loads(f.read_text())["labels"]
    items = [it for it in {json.loads(l)["uri"]: json.loads(l) for l in (D / "items.jsonl").open()}.values() if it["uri"] in labels]
    accts = sorted({it["account"] for it in items})
    random.Random(3).shuffle(accts)
    test_a = set(accts[: int(0.15 * len(accts))])
    val_a = set(accts[int(0.15 * len(accts)): int(0.22 * len(accts))])
    rt_items = list({json.loads(l)["uri"]: json.loads(l) for l in (BSKY / "data" / "reply_thread" / "items.jsonl").open()}.values())
    rt_lab = json.loads((BSKY / "data" / "reply_thread" / "labels.json").read_text())["labels"]
    return {"train": TL.recs([i for i in items if i["account"] not in test_a | val_a], labels, tok),
            "val": TL.recs([i for i in items if i["account"] in val_a], labels, tok),
            "test": TL.recs([i for i in items if i["account"] in test_a], labels, tok),
            "reply_thread": TL.recs([i for i in rt_items if i["uri"] in rt_lab], rt_lab, tok)}


EPOCHS, BS = 4, 16


def schedule(n_train):
    total = math.ceil(n_train / BS) * EPOCHS
    warm = max(1, int(0.06 * total))
    return total, warm, (lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))


def traj():
    """A seeded CPU run of thread_lora's recipe that records every step, and the fixture for the WGSL run."""
    t0 = time.time()
    m, man = model(0)
    T = man["temps"]["noul"]
    tok = TL.Enc(TL.M / "tokenizer.json")
    pad = tok.tok.token_to_id("[PAD]")
    TL.PAD = pad
    sets = splits(tok)
    out = FIX / "traj"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"T": T, "pad": pad, "epochs": EPOCHS, "bs": BS, "lr": LR, "lr_head": LR_HEAD, "wd": WD, "clip": CLIP, "sets": {}}
    for name, data in sets.items():   # ids concatenated, with offsets; positions per example; gold per option
        lens = np.array([len(d["ids"]) for d in data], np.int32)
        np.concatenate([np.array(d["ids"], np.int32) for d in data]).tofile(out / f"{name}_ids.bin")
        np.array([p for d in data for p in d["pos"]], np.int32).tofile(out / f"{name}_pos.bin")
        np.array([g for d in data for g in d["gold"]], np.float32).tofile(out / f"{name}_gold.bin")
        lens.tofile(out / f"{name}_lens.bin")
        meta["sets"][name] = {"n": len(data), "tokens": int(lens.sum()), "maxLen": int(lens.max()), "options": 2}
    flat(m).tofile(out / "params0.bin")
    total, warm, lam = schedule(len(sets["train"]))
    rng = random.Random(0)
    order = [batches(sets["train"], BS, True, rng) for _ in range(EPOCHS)]
    meta |= {"total_steps": total, "warmup": warm, "order": order, "layout": layout(m)[0], "nParams": layout(m)[1]}
    meta["evalOrder"] = {k: batches(v, BS, False, None) for k, v in sets.items() if k != "train"}
    (out / "meta.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta["sets"].items()}, {"total": total, "warm": warm, "prep_s": round(time.time() - t0)}, flush=True)

    opt = optimizer(m)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    steps, epochs, t1 = [], [], time.time()
    for ep in range(EPOCHS):
        m.train()
        for ch in order[ep]:
            _, tb = batch_arrays(sets["train"], ch, pad)
            loss = loss_of(m, tb, T)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([t for _, t in tensors(m)], CLIP)
            opt.step()
            sched.step()
            steps.append(float(loss))
            if len(steps) % 100 == 0:
                print(f"step {len(steps)} loss {np.mean(steps[-100:]):.4f} {time.time() - t1:.0f}s", flush=True)
        ev = {}
        for name in ("val", "test", "reply_thread"):
            pr = TL.p_true(m, sets[name], T)
            ev[name] = TL.metrics(pr, sets[name])
            if ep == EPOCHS - 1:
                pr.astype(np.float32).tofile(HERE / "fixtures" / f"cpu_probs_{name}.bin")
        epochs.append({"epoch": ep, "train_s": round(time.time() - t1), **ev})
        print(json.dumps(epochs[-1]), flush=True)
    res = {"steps": steps, "epochs": epochs, "train_wall_s": round(time.time() - t1), "threads": torch.get_num_threads(),
           "device": "container CPU (4 vCPU), PyTorch"}
    (HERE / "results" / "cpu_traj.json").write_text(json.dumps(res, indent=1))


def cputime():
    """Forward + backward time on the batch web/traj.js times (the middle batch of epoch 1), uncontended."""
    m, man = model(0)
    T = man["temps"]["noul"]
    tok = TL.Enc(TL.M / "tokenizer.json")
    pad = tok.tok.token_to_id("[PAD]")
    sets = splits(tok)
    meta = json.loads((FIX / "traj" / "meta.json").read_text())
    ch = meta["order"][0][len(meta["order"][0]) // 2]
    arr, tb = batch_arrays(sets["train"], ch, pad)
    ms = []
    for _ in range(6):
        t = time.perf_counter()
        loss = loss_of(m, tb, T)
        m.zero_grad()
        loss.backward()
        ms.append((time.perf_counter() - t) * 1000)
    r = {"batch": {"B": arr["B"], "S": arr["S"]}, "fwd_bwd_ms": ms[1:], "median_ms": float(np.median(ms[1:])), "threads": torch.get_num_threads()}
    (HERE / "results" / "cpu_step_time.json").write_text(json.dumps(r, indent=1))
    print(r)


if __name__ == "__main__":
    {"gradcheck": gradcheck, "traj": traj, "cputime": cputime}[sys.argv[1]]()
