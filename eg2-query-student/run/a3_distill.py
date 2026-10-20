"""A3: fine-tune a small sentence encoder so its query vectors land on EG2's.

Usage: python3 -I run/a3_distill.py --base all-MiniLM-L6-v2 [--epochs 6] [--train msmarco_train synth_train]

Student = base transformer + its own pooling + Linear(d -> 768), the linear head started
from the A1 ridge solution so epoch 0 is A1. Loss = 1 - cos(student, EG2) on the query
prompt targets, plus, with --rank-weight w > 0, w * KL(teacher || student) over softmax(score/tau)
on the index's own 2-bit scores for a doc set per batch: every batch query's teacher top-50
plus 512 random posts. Score is linear in the query, (R q) . dir * scale + bias, so the KL
term trains the student on the corpus directions that decide the ranking, which cosine alone
weights no more than any other. All weights train. Checkpoints out/a3/<base>/epoch<k>/ every epoch (resumes
from the last), and after each epoch scores the test files the same way a1_linear.py does.
Writes results/a3_<base>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_linear import TESTS, kinds, score  # noqa: E402
from common import Index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "out" / "queries"


def load_texts(name):
    return [json.loads(l)["q"] for l in (Q / f"{name}.jsonl").open()]


def build(base, X, Y):
    m = SentenceTransformer(str(ROOT / "out" / "models" / base), device="cpu")
    d = m.get_sentence_embedding_dimension()
    head = models.Dense(d, 768, bias=True, activation_function=torch.nn.Identity())
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    W = np.linalg.solve(Xc.T @ Xc + 1e-3 * len(X) * np.eye(d), Xc.T @ (Y - my))
    with torch.no_grad():
        head.linear.weight.copy_(torch.from_numpy(W.T.astype(np.float32)))
        head.linear.bias.copy_(torch.from_numpy((my - mx @ W).astype(np.float32)))
    m.append(head)
    m.append(models.Normalize())
    return m


def encode(m, texts, prompt):
    # Prompt prepended by hand, as in training, so pooling sees the same tokens either way.
    return m.encode([prompt + t for t in texts], batch_size=256, convert_to_numpy=True,
                    show_progress_bar=False, normalize_embeddings=True).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train", nargs="+", default=["msmarco_train", "synth_train"])
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--init", help="start from this checkpoint instead of base + ridge head")
    ap.add_argument("--tag", help="name for out/a3/<tag> and results/a3_<tag>.json")
    a = ap.parse_args()
    tag = a.tag or a.base + (f"_rank{a.rank_weight:g}" if a.rank_weight else "")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    texts = sum((load_texts(t) for t in a.train), [])
    Y = np.concatenate([np.load(Q / f"{t}.eg2.npy") for t in a.train])
    work = ROOT / "out" / "a3" / tag
    done = sorted(work.glob("epoch*"), key=lambda p: int(p.name[5:]))
    if done:
        m = SentenceTransformer(str(done[-1]), device="cpu")
        start = int(done[-1].name[5:]) + 1
    elif a.init:
        m = SentenceTransformer(str(ROOT / a.init), device="cpu")
        start = 0
    else:
        X = np.concatenate([np.load(ROOT / "out" / "student" / a.base / f"{t}.npy") for t in a.train])
        m = build(a.base, X, Y)
        start = 0
    base_cfg = SentenceTransformer(str(ROOT / "out" / "models" / a.base), device="cpu").prompts or {}
    prompt = base_cfg.get("query", "")
    ix = Index()
    resp = ROOT / "results" / f"a3_{tag}.json"
    res = json.loads(resp.read_text()) if resp.exists() else {"base": a.base, "train": a.train,
                                                               "n_train": len(texts), "rank_weight": a.rank_weight, "init": a.init,
                                                               "tau": a.tau, "epochs": {}}
    if a.rank_weight:
        top_p = work / "teacher_top50.npy"
        if not top_p.exists():
            work.mkdir(parents=True, exist_ok=True)
            np.save(top_p, ix.topk(Y, k=50))
        top50 = np.load(top_p)
        M = torch.from_numpy(ix.qz._rotate_rows(np.eye(768, dtype=np.float32)).astype(np.float32))
        dirs, sc, bi = (torch.from_numpy(v) for v in (ix.dirs, ix.scale, ix.bias))
        rng_docs = np.random.default_rng(1)

        def rank_loss(p, t, b):
            docs = np.unique(np.concatenate([top50[b].ravel(), rng_docs.integers(0, ix.n, 512)]))
            D, s_, b_ = dirs[docs], sc[docs], bi[docs]
            st = ((t @ M) @ D.T) * s_ + b_
            sp = ((p @ M) @ D.T) * s_ + b_
            return torch.nn.functional.kl_div(torch.log_softmax(sp / a.tau, 1),
                                              torch.log_softmax(st / a.tau, 1),
                                              log_target=True, reduction="batchmean")

    def evaluate(tag):
        r = {}
        for t in TESTS:
            if (Q / f"{t}.eg2.npy").exists():
                r[t] = score(ix, encode(m, load_texts(t), prompt), np.load(Q / f"{t}.eg2.npy"), kinds(t))
        res["epochs"][tag] = r
        resp.write_text(json.dumps(res, indent=1))
        print(tag, json.dumps({t: v["overlap@10"] for t, v in r.items()}), flush=True)

    if start == 0 and "init" not in res["epochs"]:
        evaluate("init")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(texts) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / (steps * a.epochs)))
    for _ in range(start * steps):
        sched.step()
    Yt = torch.from_numpy(Y)
    for ep in range(start, a.epochs):
        m.train()
        perm = np.random.default_rng(ep).permutation(len(texts))
        t0, tot = time.time(), 0.0
        for s in range(steps):
            b = perm[s * a.batch:(s + 1) * a.batch]
            feats = m.tokenize([prompt + texts[i] for i in b])
            out = m(feats)["sentence_embedding"]
            loss = (1 - (out * Yt[b]).sum(1)).mean()
            if a.rank_weight:
                loss = loss + a.rank_weight * rank_loss(out, Yt[b], b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if s % 200 == 0:
                print(f"ep{ep} {s}/{steps} loss {tot / (s + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        m.eval()
        m.save(str(work / f"epoch{ep}"))
        evaluate(f"epoch{ep}")


if __name__ == "__main__":
    main()
