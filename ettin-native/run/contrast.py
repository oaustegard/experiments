"""In-domain contrastive fine-tune on #102's synthetic training pairs, for any ST model (MPS).

Usage: .venv/bin/python -I run/contrast.py NAME INIT [--epochs 2] [--lr 5e-5] [--bs 64] [--qp PROMPT]

INIT = hub id or a local SentenceTransformer dir (e.g. out/students/e17-cap/epoch3).
Pairs: out/data/synth_train.jsonl (23,985 Haiku queries, each written from one post; #102 split
those posts from the test pool's 1,500, so no test post is a training positive) -> that post's
text. Loss: MultipleNegativesRankingLoss (in-batch negatives, scale 20), query prompt on the
query side. The same recipe runs on leaf-ir as the control, so an in-domain gain is not
credited to the architecture. Eval = distill.evaluate (SYM native; per-query arrays).
Writes results/contrast_NAME.json, out/contrast/NAME/epoch<k>/.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill  # noqa: E402
from screen import POSTS, Q, Index, perquery  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("init")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--qp", default=distill.QP)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    posts = [json.loads(l) for l in POSTS.open()]
    texts = [p["text"] or "" for p in posts]
    pairs = [(r["q"], texts[r["post"]]) for r in map(json.loads, (ROOT / "out" / "data" / "synth_train.jsonl").open())
             if texts[r["post"]]]
    synth = [json.loads(l) for l in (Q / "synth_test.jsonl").open()]
    assert not ({r["post"] for r in synth} & {r["post"] for r in map(json.loads, (ROOT / "out" / "data" / "synth_train.jsonl").open())})
    real = [json.loads(l) for l in (Q / "real_test.jsonl").open()]
    src = np.array([r["post"] for r in synth])
    ix = Index()
    hit_eg2 = (ix.topk(np.load(Q / "synth_test.eg2.npy")) == src[:, None]).any(1).astype(float)
    eg2_r = ix.topk(np.load(Q / "real_test.eg2.npy"))
    ctx = (texts, synth, real, src, hit_eg2, eg2_r)
    distill.QP = a.qp  # evaluate() prompts queries with distill.QP

    st = SentenceTransformer(a.init, device=dev)
    st.max_seq_length = 128
    tok = st.tokenizer
    opt = torch.optim.AdamW(st.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(pairs) / a.bs)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    res = {"args": vars(a), "n_pairs": len(pairs), "epochs": {}}
    r0 = distill.evaluate(st, None, ctx)
    perquery(f"contrast_{a.name}_ep0_native", r0.pop("native_hits"))
    res["epochs"]["0"] = r0
    print("epoch 0", json.dumps(r0), flush=True)
    rng = np.random.default_rng(0)
    enc = lambda xs: {k: v.to(dev) for k, v in tok(xs, padding=True, truncation=True, max_length=128,
                                                    return_tensors="pt").items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    st.train()
    step = 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(pairs))
        for s in range(0, len(order) - a.bs + 1, a.bs):
            b = [pairs[i] for i in order[s:s + a.bs]]
            q = torch.nn.functional.normalize(st(enc([a.qp + x for x, _ in b]))["sentence_embedding"], dim=-1)
            d = torch.nn.functional.normalize(st(enc([y for _, y in b]))["sentence_embedding"], dim=-1)
            loss = torch.nn.functional.cross_entropy(20 * q @ d.T, torch.arange(len(b), device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
        st.save(str(ROOT / "out" / "contrast" / a.name / f"epoch{ep}"))
        r = distill.evaluate(st, None, ctx) | {"epoch_s": round(time.time() - t0)}
        perquery(f"contrast_{a.name}_ep{ep}_native", r.pop("native_hits"))
        res["epochs"][str(ep)] = r
        print(f"epoch {ep}", json.dumps(r), flush=True)
        (ROOT / "results" / f"contrast_{a.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
