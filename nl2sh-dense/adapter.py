#!/usr/bin/env python3
"""Fit a query-side linear adapter on frozen embeddings, instead of fine-tuning.

`RESULTS.md` leaves the retriever at recall@3 = 0.396 and **recall@50 = 0.726**
on the same queries. A third of the eval has the gold page already in the
candidate set, ranked 4th to 50th: the retriever is finding it and ordering it
wrong. That is the condition under which training the scorer pays, and it does
not require touching the encoder.

So this trains a matrix, not a model. Query vectors get one identity-initialized
linear map `W`; document vectors stay exactly as the frozen encoder produced them,
so the 6,397 cached page vectors — and any index built from them — survive
unchanged. The cost is `d x d` floats appended to the artifact: 4.2 MB on
leaf-mt's 1024 dims, 0.6 MB on MiniLM's 384, against 25.6 MB of encoder. A
low-rank form (`--rank r`, `W = I + AB`) drops that to `2 x d x r`.

Why this before a fine-tune:

* It isolates the question. If a linear reweighting of the existing space
  recovers most of the headroom, the encoder's representation was adequate and
  the objective was the mismatch — and a fine-tune would be spending an hour of
  gradient to learn a rotation.
* It cannot damage the corpus side. A fine-tune changes every document vector,
  which means re-encoding 31k chunks per experiment and invalidating every
  cached index; this changes 164 query vectors.
* It is minutes on CPU rather than a training run, so a negative is cheap.

**Training data is NL2Bash, and the eval stays the cyber corpus** — annotator-
written English about commands, against Gemini-written English about different
commands. Nothing is shared, so this measures transfer rather than fit.
NL2Bash's 60.3% `find` skew is handled by capping pairs per utility
(`--cap`); without the cap the adapter would mostly learn to rank `find` first,
which scores well on NL2Bash and is worthless everywhere else.

Negatives are in-batch plus hard negatives mined from the frozen retriever's own
top-50 — the ranks this exists to fix. An adapter trained on random negatives
learns to separate `tar` from `ping`, which the base encoder already does.

    python3 adapter.py train --model leaf-mt-int8 --granularity page \\
        --nl2bash <nl2bash>/data/bash
    python3 eval_dense.py --models leaf-mt-int8 --granularity page \\
        --adapter cache/adapter_leaf-mt-int8_page.npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
sys.path.insert(0, str(RETRIEVAL))
import pleias_gate as G  # noqa: E402
import retrieve as R  # noqa: E402
import dense_index as D  # noqa: E402
import encoders  # noqa: E402


def training_pairs(nl2bash: Path, utilities: set[str], cap: int, seed: int,
                   limit: int) -> list[tuple[str, str]]:
    """(request, gold utility) from NL2Bash, capped per utility to kill the skew."""
    nls = (nl2bash / "all.nl").read_text(errors="replace").splitlines()
    cms = (nl2bash / "all.cm").read_text(errors="replace").splitlines()
    rng = random.Random(seed)
    pool = []
    for nl, cm in zip(nls, cms):
        u = G.gold_utility(cm)
        if u in utilities and nl.strip():
            pool.append((nl.strip(), u))
    rng.shuffle(pool)
    seen: dict[str, int] = {}
    out = []
    for nl, u in pool:
        if seen.get(u, 0) >= cap:
            continue
        seen[u] = seen.get(u, 0) + 1
        out.append((nl, u))
    return out[:limit]


def mine_hard_negatives(qv: np.ndarray, doc_vectors: np.ndarray,
                        doc_utilities: np.ndarray, golds: list[str],
                        depth: int, per_query: int, rng: random.Random) -> np.ndarray:
    """Document indices the frozen retriever ranks highly and that are not gold.

    Sampled from the top `depth` rather than taken from the top `per_query`,
    because the hardest negatives are often near-duplicates of the gold page and
    training against them teaches the adapter to split hairs it will never be
    asked about. Sampling spreads the pressure over the band the eval actually
    loses in.
    """
    out = np.empty((len(qv), per_query), dtype=np.int32)
    for i in range(len(qv)):
        s = doc_vectors @ qv[i]
        cand = np.argpartition(-s, depth)[:depth]
        cand = [int(j) for j in cand if doc_utilities[j] != golds[i]]
        if len(cand) < per_query:
            cand += [int(rng.randrange(len(doc_vectors))) for _ in range(per_query)]
        out[i] = rng.sample(cand, per_query)
    return out


def train(a) -> int:
    import torch

    chunks = D.GRANULARITIES[a.granularity](R.load_chunks(a.chunks))
    doc_utilities = np.array([c.utility for c in chunks], dtype=object)
    doc_vectors = D.build_vectors(a.model, chunks, granularity=a.granularity)
    utilities = set(doc_utilities)

    # One document per utility — the best-scoring page is what the eval ranks,
    # so the positive is that utility's own page rather than an arbitrary chunk.
    first_doc: dict[str, int] = {}
    for i, u in enumerate(doc_utilities):
        first_doc.setdefault(str(u), i)

    pairs = training_pairs(a.nl2bash, utilities, a.cap, a.seed, a.limit)
    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * a.val_frac))
    val, tr = pairs[:n_val], pairs[n_val:]
    print(f"{len(pairs)} pairs over {len(set(u for _, u in pairs))} utilities "
          f"(cap {a.cap}); {len(tr)} train / {len(val)} val", file=sys.stderr)

    enc = encoders.build(a.model)
    t0 = time.time()
    qv_all = enc.encode([nl for nl, _ in pairs], prompt="query",
                        batch_size=32, progress=True)
    print(f"encoded {len(pairs)} queries in {time.time() - t0:.0f}s", file=sys.stderr)
    qv_tr, qv_val = qv_all[n_val:], qv_all[:n_val]
    pos_tr = np.array([first_doc[u] for _, u in tr], dtype=np.int64)

    negs = mine_hard_negatives(qv_tr, doc_vectors, doc_utilities,
                               [u for _, u in tr], a.neg_depth, a.negatives, rng)
    print(f"mined {a.negatives} hard negatives per query from the top "
          f"{a.neg_depth}", file=sys.stderr)

    dim = qv_all.shape[1]
    Dv = torch.from_numpy(doc_vectors.astype(np.float32))
    Q = torch.from_numpy(qv_tr.astype(np.float32))
    P = torch.from_numpy(pos_tr)
    N = torch.from_numpy(negs.astype(np.int64))

    if a.rank:
        A = torch.zeros(dim, a.rank, requires_grad=True)
        B = torch.zeros(a.rank, dim, requires_grad=True)
        torch.nn.init.normal_(A, std=0.02)
        params = [A, B]
        def W():
            return torch.eye(dim) + A @ B
    else:
        Wd = torch.zeros(dim, dim, requires_grad=True)
        params = [Wd]
        def W():
            return torch.eye(dim) + Wd

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)

    def score_val(mat) -> float:
        """recall@3 over utilities, on the held-out NL2Bash slice."""
        with torch.no_grad():
            q = torch.from_numpy(qv_val.astype(np.float32)) @ mat.T
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
            s = q @ Dv.T
            top = torch.topk(s, 30, dim=1).indices.numpy()
        hit = 0
        for row, (_, gold) in zip(top, val):
            seen = []
            for j in row:
                u = str(doc_utilities[j])
                if u not in seen:
                    seen.append(u)
                if len(seen) >= 3:
                    break
            hit += gold in seen
        return hit / len(val)

    with torch.no_grad():
        base = score_val(torch.eye(dim))
    print(f"held-out recall@3 before training: {base:.3f}", file=sys.stderr)

    idx = np.arange(len(tr))
    best, best_state, t0 = base, None, time.time()
    for ep in range(a.epochs):
        rng.shuffle(idx)
        total = 0.0
        for s in range(0, len(idx), a.batch):
            sel = idx[s : s + a.batch]
            if len(sel) < 2:
                continue
            mat = W()
            q = Q[sel] @ mat.T
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
            # Candidates: this batch's positives (in-batch negatives for each
            # other) followed by every hard negative mined for the batch.
            cand = torch.cat([P[sel], N[sel].reshape(-1)])
            logits = (q @ Dv[cand].T) / a.temperature
            loss = torch.nn.functional.cross_entropy(
                logits, torch.arange(len(sel)))
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(loss.detach()) * len(sel)
        with torch.no_grad():
            mat = W().detach()
            v = score_val(mat)
        print(f"  ep{ep} loss {total / len(idx):.4f}  held-out recall@3 {v:.3f}  "
              f"{(time.time() - t0) / 60:.1f}m", flush=True)
        if v > best:
            best, best_state = v, mat.numpy().copy()

    if best_state is None:
        print("no epoch beat the identity on held-out data; not saving",
              file=sys.stderr)
        return 1
    out = a.out or (D.CACHE / f"adapter_{a.model}_{a.granularity}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=best_state.astype(np.float32),
             meta=json.dumps({"model": a.model, "granularity": a.granularity,
                              "rank": a.rank, "pairs": len(pairs), "cap": a.cap,
                              "epochs": a.epochs, "temperature": a.temperature,
                              "negatives": a.negatives, "neg_depth": a.neg_depth,
                              "val_recall3_base": round(base, 3),
                              "val_recall3_best": round(best, 3)}))
    print(f"saved {out.name}: held-out recall@3 {base:.3f} -> {best:.3f}, "
          f"{best_state.nbytes / 1e6:.1f} MB")
    return 0


def load_adapter(path: Path) -> np.ndarray:
    return np.load(path)["W"]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default="leaf-mt-int8", choices=sorted(encoders.ENCODERS))
    t.add_argument("--granularity", default="page", choices=sorted(D.GRANULARITIES))
    t.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    t.add_argument("--nl2bash", type=Path, required=True)
    t.add_argument("--cap", type=int, default=20, help="max pairs per gold utility")
    t.add_argument("--limit", type=int, default=100000)
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--rank", type=int, default=0, help="0 = full d x d, else low-rank")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-2)
    t.add_argument("--temperature", type=float, default=0.05)
    t.add_argument("--negatives", type=int, default=4)
    t.add_argument("--neg-depth", type=int, default=50)
    t.add_argument("--seed", type=int, default=20260820)
    t.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())
