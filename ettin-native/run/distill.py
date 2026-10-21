"""LEAF-style distillation of an Ettin encoder into a teacher's embedding space (MPS).

Usage: .venv/bin/python -I run/distill.py NAME --base jhu-clsp/ettin-encoder-17m \
         [--teacher snowflake-arctic-embed-m-v1.5] [--epochs 4] [--lr 3e-4] [--bs 128]

Student = sentence-transformers [Transformer(base, 128 tokens), mean Pooling, Dense(H -> D,
no bias, identity)], D = teacher dim. Loss = 1 - cos(student, teacher) on the rows of
out/data/train.jsonl (run/teacher.py). Same prompts as the teacher (query prompt on role q,
none on docs), so the student is query/doc aware the way leaf-ir is. No labels, no negatives.
After each epoch: hit@10 on the Bluesky task natively (student docs + queries, fp32) and
asymmetrically (student queries vs teacher docs, the LEAF deployment), saved under
out/students/NAME/epoch<k>/ as a SentenceTransformer directory that run/export.py loads.
Writes results/distill_NAME.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen import POSTS, Q, Index, boot_ci, native, overlap, perquery  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "out" / "data"
QP = "Represent this sentence for searching relevant passages: "


def evaluate(st, teacher_docs, ctx):
    texts, synth, real, src, hit_eg2, eg2_r = ctx
    st.eval()
    with torch.no_grad():
        X = st.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        Qs = st.encode([QP + r["q"] for r in synth], batch_size=256, normalize_embeddings=True)
        Qr = st.encode([QP + r["q"] for r in real], batch_size=256, normalize_embeddings=True)
    out = {}
    for name, docs in (("native", X), ("asym", teacher_docs)):
        if docs is None:
            continue
        (ts, _), (tr, _) = native.rx2_topk(docs, Qs), native.rx2_topk(docs, Qr)
        h = (ts == src[:, None]).any(1).astype(float)
        out[name + "_hits"] = h
        out[name] = {"hit@10": boot_ci(h), "minus_eg2": native.paired(h, hit_eg2), "hit@1": float((ts[:, 0] == src).mean()),
                     "real_overlap_with_eg2": boot_ci(overlap(tr, eg2_r))}
    if teacher_docs is not None:
        out["doc_cos_to_teacher"] = float((X * teacher_docs).sum(1).mean())
    st.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", default="jhu-clsp/ettin-encoder-17m")
    ap.add_argument("--teacher", default="snowflake-arctic-embed-m-v1.5")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--no-posts", action="store_true", help="drop the corpus posts from training (generality control)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [json.loads(l) for l in (D / "train.jsonl").open()]
    T = np.load(D / f"{a.teacher}.f16.npy")
    posts = [json.loads(l) for l in POSTS.open()]
    texts = [p["text"] or "" for p in posts]
    keep = np.arange(len(rows))
    if a.no_posts:
        ps = {t for t in texts if t}
        keep = np.array([i for i, r in enumerate(rows) if not (r["role"] == "d" and r["t"] in ps)])
    synth = [json.loads(l) for l in (Q / "synth_test.jsonl").open()]
    real = [json.loads(l) for l in (Q / "real_test.jsonl").open()]
    src = np.array([r["post"] for r in synth])
    ix = Index()
    hit_eg2 = (ix.topk(np.load(Q / "synth_test.eg2.npy")) == src[:, None]).any(1).astype(float)
    eg2_r = ix.topk(np.load(Q / "real_test.eg2.npy"))
    ctx = (texts, synth, real, src, hit_eg2, eg2_r)
    tdoc_path = ROOT / "out" / "screen" / f"Snowflake__{a.teacher}" / "docs.npy"
    teacher_docs = np.load(tdoc_path) if tdoc_path.exists() else None

    tr = models.Transformer(a.base, max_seq_length=128)
    H = tr.get_word_embedding_dimension()
    st = SentenceTransformer(modules=[tr, models.Pooling(H, "mean"),
                                      models.Dense(H, T.shape[1], bias=False, activation_function=torch.nn.Identity())],
                             device=dev)
    st.prompts = {"query": QP, "document": ""}
    tok = st.tokenizer
    opt = torch.optim.AdamW(st.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(keep) / a.bs)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    res_path = ROOT / "results" / f"distill_{a.name}.json"
    res = {"args": vars(a), "n_train": int(len(keep)), "params_m": round(sum(p.numel() for p in st.parameters()) / 1e6, 2), "epochs": {}}
    res["epochs"]["0"] = {k: v for k, v in evaluate(st, teacher_docs, ctx).items() if not k.endswith("_hits")}
    print("epoch 0", json.dumps(res["epochs"]["0"]), flush=True)
    rng = np.random.default_rng(0)
    lens = [len(r["t"]) for r in rows]
    step = 0
    st.train()
    for ep in range(1, a.epochs + 1):
        # length-bucketed batches (cloud lesson, #104): sort shuffled chunks of 50 batches by
        # character length, cut into batches, shuffle the batches
        perm = rng.permutation(keep)
        chunk = 50 * a.bs
        batches = []
        for c0 in range(0, len(perm), chunk):
            part = perm[c0:c0 + chunk]
            part = part[np.argsort([lens[i] for i in part], kind="stable")]
            batches += [part[j:j + a.bs] for j in range(0, len(part), a.bs)]
        batches = [batches[i] for i in rng.permutation(len(batches))]
        t0, run = time.time(), 0.0
        for s, b in enumerate(batches):
            s *= a.bs
            batch = [(QP if rows[i]["role"] == "q" else "") + rows[i]["t"] for i in b]
            f = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            f = {k: v.to(dev) for k, v in f.items() if k in ("input_ids", "attention_mask")}
            e = st(f)["sentence_embedding"]
            t = torch.from_numpy(T[b].astype(np.float32)).to(dev)
            loss = (1 - torch.nn.functional.cosine_similarity(e, t)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss {run:.4f} {(time.time() - t0) / (s // a.bs + 1):.2f} s/step", flush=True)
        out = ROOT / "out" / "students" / a.name / f"epoch{ep}"
        st.save(str(out))
        r = evaluate(st, teacher_docs, ctx) | {"loss": run, "epoch_s": round(time.time() - t0)}
        for k in [k for k in r if k.endswith("_hits")]:
            perquery(f"distill_{a.name}_ep{ep}_{k[:-5]}", r.pop(k))
        res["epochs"][str(ep)] = r
        print(f"epoch {ep}", json.dumps(r), flush=True)
        res_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
