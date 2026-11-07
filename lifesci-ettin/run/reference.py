"""PyTorch fixtures for web/distill.js.

    python3 -I run/reference.py gradcheck   # fixtures/gradcheck/: one batch, loss, every gradient, one AdamW step
    python3 -I run/reference.py f64 RESULT  # compare a gradcheck result's raw grads against float64
    python3 -I run/reference.py nce         # fixtures/gradcheck_nce/: the same for the contrastive loss (_lib/wgsl/train/nce.js)
    python3 -I run/reference.py f64nce RESULT

The gradcheck model is e17c with its table cut to the batch's tokens plus random rows (V = 2048), so the
fixture stays small; ids are remapped to that table. Teacher targets are random: the check is the math.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import Embedder, layout, load_st, to_flat  # noqa: E402

FIX = HERE / "fixtures" / "gradcheck"
LR, WD, CLIP, V = 1e-4, 0.01, 1.0, 2048
torch.set_num_threads(4)


def fixture():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HERE / ".work/e17c/epoch1/tokenizer.json"))
    docs = [json.loads(l)["text"] for l in open(HERE.parent / "lifesci-vocab-probe/data/abstracts_2019.jsonl")][:400]
    enc = [tok.encode(d).ids for d in docs]
    want = [20, 60, 130, 150, 200, 260, 300, 400]
    stream = [t for e in enc for t in e]
    seqs, o = [], 0
    for w in want:
        seqs.append(stream[o:o + w])
        o += w
    uniq = sorted({t for s in seqs for t in s})
    assert len(uniq) < V
    p = load_st(HERE / ".work/e17c/epoch1")
    g = torch.Generator().manual_seed(0)
    extra = torch.randint(0, p["table"].shape[0], (V - len(uniq),), generator=g)
    p["table"] = torch.cat([p["table"][uniq], p["table"][extra]])
    remap = {t: i for i, t in enumerate(uniq)}
    pad = len(uniq)  # any row; padding rows carry no gradient
    S, B = max(want), len(want)
    ids = torch.full((B, S), pad, dtype=torch.long)
    att = torch.zeros((B, S), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor([remap[t] for t in s])
        att[b, :len(s)] = 1
    tgt = torch.randn((B, 768), generator=g)
    return p, ids, att, tgt


def gradcheck():
    p, ids, att, tgt = fixture()
    m = Embedder(p)
    p0 = to_flat(m)
    y = m(ids, att)
    loss = (1 - torch.nn.functional.cosine_similarity(y, tgt)).mean()
    loss.backward()
    ps = [t for _, t in m.ordered()]
    g = torch.cat([t.grad.reshape(-1) for t in ps]).numpy().astype(np.float32)
    norm = float(torch.nn.utils.clip_grad_norm_(ps, CLIP))
    opt = torch.optim.AdamW(ps, lr=LR, weight_decay=WD)
    opt.step()
    p1 = to_flat(m)
    FIX.mkdir(parents=True, exist_ok=True)
    ids.numpy().astype(np.int32).reshape(-1).tofile(FIX / "ids.bin")
    att.sum(1).numpy().astype(np.int32).tofile(FIX / "lens.bin")
    tgt.numpy().astype(np.float32).tofile(FIX / "tgt.bin")
    p0.tofile(FIX / "params0.bin"); g.tofile(FIX / "grads.bin"); p1.tofile(FIX / "params1.bin")
    lay, n = layout(V)
    meta = {"B": ids.shape[0], "S": ids.shape[1], "V": V, "loss": float(loss), "grad_norm": norm, "layout": lay, "nParams": n,
            "lens": att.sum(1).tolist(), "adam": {"lr": LR, "wd": WD, "clip": CLIP, "step": 1}}
    (FIX / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: meta[k] for k in ("B", "S", "loss", "grad_norm", "nParams", "lens")})


def f64(result):
    r = json.loads(Path(result).read_text())
    p, ids, att, tgt = fixture()
    m = Embedder(p).double()
    y = m(ids, att)
    loss = (1 - torch.nn.functional.cosine_similarity(y, tgt.double())).mean()
    loss.backward()
    g64 = torch.cat([t.grad.reshape(-1) for _, t in m.ordered()]).numpy()
    g32 = np.fromfile(FIX / "grads.bin", np.float32)
    gg = np.asarray(r["grads"], np.float64)
    lay, _ = layout(V)
    out = {"loss64": float(loss), "loss_gpu": r["loss"], "loss_torch32": json.loads((FIX / "meta.json").read_text())["loss"], "tensors": []}
    tot = np.linalg.norm(g64)
    for e in lay:
        n = int(np.prod(e["shape"]))
        s = slice(e["offset"], e["offset"] + n)
        den = np.linalg.norm(g64[s])
        if den < 1e-9 * tot:
            continue
        out["tensors"].append({"name": e["name"], "gpu_vs_64": float(np.linalg.norm(gg[s] - g64[s]) / den),
                               "torch32_vs_64": float(np.linalg.norm(g32[s] - g64[s]) / den)})
    out["max_gpu_vs_64"] = max(t["gpu_vs_64"] for t in out["tensors"])
    out["max_torch32_vs_64"] = max(t["torch32_vs_64"] for t in out["tensors"])
    print(json.dumps({k: v for k, v in out.items() if k != "tensors"}, indent=1))
    return out


NCE_P, SCALE = 8, 20.0
FIXN = HERE / "fixtures" / "gradcheck_nce"


def nce_loss(y, P, scale=SCALE):
    """MNRL: rows 0..P-1 queries, P..2P-1 their positives; mean_i CE(scale * cos(q_i, d_j), i)."""
    yn = torch.nn.functional.normalize(y, dim=-1)
    sim = scale * yn[:P] @ yn[P:].T
    return torch.nn.functional.cross_entropy(sim, torch.arange(P))


def fixture_nce():
    """8 query/passage pairs from the training shards (run/pairs.py order), stock tokenizer."""
    S_ = HERE / ".work/shards"
    lens = np.fromfile(S_ / "stock.lens.u16", np.uint16).astype(np.int64)
    ids = np.fromfile(S_ / "stock.ids.u16", np.uint16)
    offs = np.concatenate([[0], np.cumsum(lens)])
    pr = np.load(S_ / "pairs.npz")["order"][:NCE_P]
    rows = list(pr[:, 0]) + list(pr[:, 1])
    seqs = [ids[offs[r]:offs[r + 1]].astype(np.int64).tolist() for r in rows]
    uniq = sorted({t for s in seqs for t in s})
    p = load_st(HERE / ".work/e17c/epoch1")
    g = torch.Generator().manual_seed(0)
    extra = torch.randint(0, p["table"].shape[0], (V - len(uniq),), generator=g)
    p["table"] = torch.cat([p["table"][uniq], p["table"][extra]])
    remap = {t: i for i, t in enumerate(uniq)}
    S, B = max(len(s) for s in seqs), len(seqs)
    idt = torch.full((B, S), len(uniq), dtype=torch.long)
    att = torch.zeros((B, S), dtype=torch.long)
    for b, s in enumerate(seqs):
        idt[b, :len(s)] = torch.tensor([remap[t] for t in s])
        att[b, :len(s)] = 1
    return p, idt, att


def gradcheck_nce():
    p, ids, att = fixture_nce()
    m = Embedder(p)
    p0 = to_flat(m)
    loss = nce_loss(m(ids, att), NCE_P)
    loss.backward()
    ps = [t for _, t in m.ordered()]
    g = torch.cat([t.grad.reshape(-1) for t in ps]).numpy().astype(np.float32)
    norm = float(torch.nn.utils.clip_grad_norm_(ps, CLIP))
    torch.optim.AdamW(ps, lr=LR, weight_decay=WD).step()
    p1 = to_flat(m)
    FIXN.mkdir(parents=True, exist_ok=True)
    ids.numpy().astype(np.int32).reshape(-1).tofile(FIXN / "ids.bin")
    att.sum(1).numpy().astype(np.int32).tofile(FIXN / "lens.bin")
    p0.tofile(FIXN / "params0.bin"); g.tofile(FIXN / "grads.bin"); p1.tofile(FIXN / "params1.bin")
    lay, n = layout(V)
    meta = {"B": ids.shape[0], "S": ids.shape[1], "V": V, "loss": float(loss), "grad_norm": norm, "layout": lay, "nParams": n,
            "lens": att.sum(1).tolist(), "adam": {"lr": LR, "wd": WD, "clip": CLIP, "step": 1}, "loss_kind": "nce", "scale": SCALE}
    (FIXN / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: meta[k] for k in ("B", "S", "loss", "grad_norm", "nParams", "lens")})


def f64nce(result):
    r = json.loads(Path(result).read_text())
    p, ids, att = fixture_nce()
    m = Embedder(p).double()
    loss = nce_loss(m(ids, att), NCE_P)
    loss.backward()
    g64 = torch.cat([t.grad.reshape(-1) for _, t in m.ordered()]).numpy()
    g32 = np.fromfile(FIXN / "grads.bin", np.float32)
    gg = np.asarray(r["grads"], np.float64)
    lay, _ = layout(V)
    out = {"loss64": float(loss), "loss_gpu": r["loss"], "tensors": []}
    tot = np.linalg.norm(g64)
    for e in lay:
        n = int(np.prod(e["shape"]))
        s = slice(e["offset"], e["offset"] + n)
        den = np.linalg.norm(g64[s])
        if den < 1e-9 * tot:
            continue
        out["tensors"].append({"name": e["name"], "gpu_vs_64": float(np.linalg.norm(gg[s] - g64[s]) / den),
                               "torch32_vs_64": float(np.linalg.norm(g32[s] - g64[s]) / den)})
    out["max_gpu_vs_64"] = max(t["gpu_vs_64"] for t in out["tensors"])
    out["max_torch32_vs_64"] = max(t["torch32_vs_64"] for t in out["tensors"])
    out["update_rel_err"] = r.get("updateRelErr")
    print(json.dumps({k: v for k, v in out.items() if k != "tensors"}, indent=1))
    return out


if __name__ == "__main__":
    {"gradcheck": gradcheck, "f64": lambda: f64(sys.argv[2]), "nce": gradcheck_nce,
     "f64nce": lambda: (HERE / "results" / "gradcheck_nce_f64.json").write_text(json.dumps(f64nce(sys.argv[2]), indent=1))}[sys.argv[1]]()
