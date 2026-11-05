"""A LoRA spike for the per-post "about AI" question, trained on thread-aware labels against the shipped base.

    python3 -I run/thread_lora.py parity    # torch model rebuilt from web/models/account vs the NumPy reference
    python3 -I run/thread_lora.py train     # -> out/thread_lora/adapter.pt, results/thread_lora.json

The base is the packed page model itself, dequantized (int8 matmuls, 4-bit token table), so the
adapter is trained against exactly what the browser computes; the original fp32 checkpoint is not
needed. LoRA follows adapter-library (rank 8, alpha 16, on Wqkv / Wo / Wi; the head is trained too).
Only the per-post inputs would use the adapter: the account questions keep the base.

Data: data/thread_lora (run/thread_data.py), split by account, 15% of accounts held out. A second
test set is data/reply_thread, the 376 replies of the whole-thread round, from accounts not in the
training data.
"""
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
BD = ROOT.parent / "browser-decider" / "run"
sys.path[:0] = [str(ROOT / "run"), str(BD)]
from build_data import POST_Q  # noqa: E402
from common import examples  # noqa: E402
from pack import Ref  # noqa: E402
from state import post_state  # noqa: E402
from train_enc import Decider, Enc, batches, collate, predict, seg_softmax  # noqa: E402

M = ROOT / "web" / "models" / "account"
OUT = ROOT / "out" / "thread_lora"
RANK, ALPHA, TARGETS, LR = 8, 16, {"Wqkv", "Wo", "Wi"}, 5e-4
torch.set_num_threads(4)


def stock_path():
    from huggingface_hub import snapshot_download
    return snapshot_download("jhu-clsp/ettin-encoder-17m", allow_patterns=["*.json", "*.bin", "*.safetensors"])


def base_model():
    """Decider whose weights are the dequantized packed page model."""
    man = json.loads((M / "manifest.json").read_text())
    ref = Ref(man, (M / "weights.bin").read_bytes())
    t = lambda r, dt=np.float32: torch.from_numpy(ref.t(r, dt).astype(np.float32).copy())  # noqa: E731
    lin = lambda w: torch.from_numpy((ref.t(w["q"], np.int8).astype(np.float32) * ref.t(w["s"])).T.copy())  # noqa: E731
    T = man["table"]
    p = ref.t(T["codes"], np.uint8)
    q = np.stack([p & 15, p >> 4], -1).reshape(T["V"], T["H"]).astype(np.float32) - 8
    s = ref.t(T["scales"], np.float16).astype(np.float32).reshape(T["V"], T["H"] // 32, 1)
    sd = {"enc.embeddings.tok_embeddings.weight": torch.from_numpy((q.reshape(T["V"], -1, 32) * s).reshape(T["V"], T["H"])),
          "enc.embeddings.norm.weight": t(man["emb_norm"]), "enc.final_norm.weight": t(man["final_norm"])}
    for i, L in enumerate(man["layers"]):
        pf = f"enc.layers.{i}."
        if L["attn_norm"] is not None:
            sd[pf + "attn_norm.weight"] = t(L["attn_norm"])
        sd |= {pf + "attn.Wqkv.weight": lin(L["Wqkv"]), pf + "attn.Wo.weight": lin(L["Wo"]),
               pf + "mlp_norm.weight": t(L["mlp_norm"]), pf + "mlp.Wi.weight": lin(L["Wi"]), pf + "mlp.Wo.weight": lin(L["Wo_mlp"])}
    hd = man["head"]
    sd |= {"head.0.weight": t(hd["w0"]), "head.0.bias": t(hd["b0"]), "head.2.weight": t(hd["ln_w"]),
           "head.2.bias": t(hd["ln_b"]), "head.3.weight": t(hd["w1"]).reshape(1, -1), "head.3.bias": torch.tensor([hd["b1"]])}
    m = Decider(stock_path())
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if "rotary" not in k], (missing, unexpected)
    return m, man, ref


class LoRALinear(nn.Module):
    def __init__(self, base, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(m):
    for p in m.enc.parameters():
        p.requires_grad_(False)
    for _, mod in list(m.enc.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and cn in TARGETS:
                setattr(mod, cn, LoRALinear(ch))
    return m


def recs(items, labels, tok):
    out = []
    for it in items:
        p = labels[it["uri"]]
        e = examples([{"id": it["uri"], "workflow": "post", "state": post_state(it), "questions": POST_Q,
                       "gold": {"about_ai": {"type": "noul", "probabilities": {"true": p, "false": 1 - p}}}}])[0]
        ids, pos = tok.build(e)
        out.append({**e, "ids": ids, "pos": pos, "kind": it["kind"], "source": it.get("source")})
    return out


def p_true(model, data, T):
    lg = predict(model, data, PAD)
    return np.array([float(np.exp(z[0] / T) / np.exp(np.asarray(z) / T).sum()) for z in lg])  # option 0 is "true"


def metrics(pred, data):
    g = np.array([d["gold"][0] for d in data])
    pos = g > 0.5
    out = {"n": len(data), "accuracy": round(float(((pred > 0.5) == pos).mean()), 4),
           "recall": round(float((pred[pos] > 0.5).mean()), 4) if pos.any() else None,
           "false_pos_rate": round(float((pred[~pos] > 0.5).mean()), 4) if (~pos).any() else None,
           "mean_abs_err": round(float(np.abs(pred - g).mean()), 4)}
    for k in ("reply", "post", "repost"):
        sel = np.array([d["kind"] == k for d in data])
        if sel.any() and len(data) > sel.sum():
            out[k] = {"n": int(sel.sum()), "accuracy": round(float(((pred[sel] > 0.5) == pos[sel]).mean()), 4)}
    return out


def parity():
    m, man, ref = base_model()
    tok = Enc(M / "tokenizer.json")
    global PAD
    PAD = tok.tok.token_to_id("[PAD]")
    items = [json.loads(l) for l in (ROOT / "data" / "reply_thread" / "items.jsonl").open()][:20]
    data = recs(items, {it["uri"]: 0.5 for it in items}, tok)
    lg = predict(m, data, PAD)
    d = max(float(np.abs(np.asarray(z) - ref.logits(e["ids"], e["pos"])).max()) for z, e in zip(lg, data))
    print({"max_abs_logit_diff_vs_numpy_ref": d})


def train():
    t0 = time.time()
    m, man, _ = base_model()
    T = man["temps"]["noul"]
    tok = Enc(M / "tokenizer.json")
    global PAD
    PAD = tok.tok.token_to_id("[PAD]")
    D = ROOT / "data" / "thread_lora"
    labels = {}
    for f in sorted((D / "labels").glob("b*.json")):
        labels |= json.loads(f.read_text())["labels"]
    items = [it for it in {json.loads(l)["uri"]: json.loads(l) for l in (D / "items.jsonl").open()}.values() if it["uri"] in labels]
    accts = sorted({it["account"] for it in items})
    random.Random(3).shuffle(accts)
    test_a = set(accts[: int(0.15 * len(accts))])
    val_a = set(accts[int(0.15 * len(accts)): int(0.22 * len(accts))])
    tr = recs([i for i in items if i["account"] not in test_a | val_a], labels, tok)
    va = recs([i for i in items if i["account"] in val_a], labels, tok)
    te = recs([i for i in items if i["account"] in test_a], labels, tok)
    rt_items = list({json.loads(l)["uri"]: json.loads(l) for l in (ROOT / "data" / "reply_thread" / "items.jsonl").open()}.values())
    rt_lab = json.loads((ROOT / "data" / "reply_thread" / "labels.json").read_text())["labels"]
    rt = recs([i for i in rt_items if i["uri"] in rt_lab], rt_lab, tok)
    print({"train": len(tr), "val": len(va), "test": len(te), "reply_thread": len(rt), "load_s": round(time.time() - t0)}, flush=True)

    res = {"base": {"test": metrics(p_true(m, te, T), te), "reply_thread": metrics(p_true(m, rt, T), rt)}}
    print("base", json.dumps(res["base"]), flush=True)
    add_lora(m)
    body = [p for p in m.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": LR}, {"params": list(m.head.parameters()), "lr": LR * 1.6}], weight_decay=0.01)
    epochs, bs = 4, 16
    total = math.ceil(len(tr) / bs) * epochs
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    rng, best, log = random.Random(0), (float("inf"), -1, None), []
    for ep in range(epochs):
        m.train()
        for k, ch in enumerate(batches(tr, bs, True, rng)):
            ids, att, pb, pt, seg, gold = collate([tr[i] for i in ch], PAD)
            z = m(ids, att, pb, pt) / T
            loss = -(gold * torch.cat(seg_softmax(z, seg))).sum() / len(seg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            if k % 100 == 0:
                print(f"epoch {ep} step {k} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        pv = p_true(m, va, T)
        g = np.array([d["gold"][0] for d in va])
        vce = float(-(g * np.log(pv + 1e-9) + (1 - g) * np.log(1 - pv + 1e-9)).mean())
        log.append({"epoch": ep, "val_ce": round(vce, 4), "val": metrics(pv, va), "s": round(time.time() - t0)})
        print(json.dumps(log[-1]), flush=True)
        if vce < best[0]:
            best = (vce, ep, {k: v.detach().clone() for k, v in m.state_dict().items() if ".A" in k or ".B" in k or k.startswith("head.")})
    m.load_state_dict(best[2], strict=False)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], OUT / "adapter.pt")
    res["lora"] = {"test": metrics(p_true(m, te, T), te), "reply_thread": metrics(p_true(m, rt, T), rt)}
    res |= {"best_epoch": best[1], "log": log, "counts": {"train": len(tr), "val": len(va), "test": len(te), "reply_thread": len(rt)},
            "rank": RANK, "targets": sorted(TARGETS), "wall_s": round(time.time() - t0),
            "trainable_params": sum(p.numel() for p in best[2].values())}
    (ROOT / "results" / "thread_lora.json").write_text(json.dumps(res, indent=1))
    print("lora", json.dumps(res["lora"]), flush=True)


PAD = None
if __name__ == "__main__":
    {"parity": parity, "train": train}[sys.argv[1]]()
