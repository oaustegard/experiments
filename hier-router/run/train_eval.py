"""GPU part: train the flat router and the tree router, then score every routing configuration.

    python3 -I run/train_eval.py [--smoke]

Both routers are zero-start LoRA decision models on frozen stock Ettin-17m (adapter-library lora.py
recipe: rank 8 on Wqkv/Wo/Wi, MLP head, AdamW 5e-4), trained on the same 62 leaves' train cases.

  flat   one question per case with the gold leaf among a random subset of 3..61 other leaves, plus
         `none`; three subsets per case. 15% of examples drop the gold leaf and are labelled `none`.
  tree_body  the tree router's data and questions, but every layer trained and the token table frozen
         (LR 8e-5, lora.py's full fine-tune rate); answers whether a whole-body adapter routes better.
  tree   one shared adapter for every router node. Per case: one question at each node on its path,
         options = the node's children (each non-gold sibling dropped with p 0.25) plus `none`; plus one
         question at a router node off its path (a sibling of a path node when there is one), gold `none`.

Eval writes out/eval.json: for every configuration and test case, the probabilities each router call
returned, enough for analyze.py to compute greedy and beam routes on CPU. Sequences are capped at
2048 tokens (no state is cut: options plus state stay under 1,700 tokens at 62 leaves).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree as T  # noqa: E402  (puts adapter-library/run on sys.path)
import train_enc  # noqa: E402
from lora import adapter_state, load_adapter, new_model, train  # noqa: E402
from train_enc import Enc, predict, probs_at  # noqa: E402

train_enc.MAX_LEN = 2048
ROOT = T.ROOT
OUT = ROOT / "out"
SIZES, DRAWS, SHARD = [4, 8, 16, 32, 62], 3, 16
EPOCHS = 3


def tok_examples(tok, exs):
    for e in exs:
        e["ids"], e["pos"] = tok.build(e)
    return exs


def flat_train(cases, rng):
    out = []
    for c in cases:
        others = [x for x in T.TRAINED if x != c["leaf"]]
        for _ in range(3):
            opts = rng.sample(others, rng.randint(3, len(others)))
            gold = c["leaf"]
            if rng.random() < 0.15:
                gold = "none"
            else:
                opts.append(c["leaf"])
            rng.shuffle(opts)
            out.append(T.router_example("flat", opts, c["state"], gold, c["id"]))
    return out


def tree_train(cases, rng):
    kids = T.induced(T.TRAINED)
    routers = list(kids)
    out = []
    for c in cases:
        path = [n for n in T.path_of(c["leaf"]) if n in kids or n == c["leaf"]]
        # the induced tree can skip collapsed routers; walk it instead of the full path
        node, onpath = "root", ["root"]
        while node in kids:
            nxt = next(k for k in kids[node] if k == c["leaf"] or (k in kids and c["leaf"] in subtree(kids, k)))
            opts = [k for k in kids[node] if k == nxt or rng.random() >= 0.25]
            rng.shuffle(opts)
            out.append(T.router_example(node, opts, c["state"], nxt, c["id"]))
            node = nxt
            onpath.append(node)
        off = [r for r in routers if r not in onpath]
        sib = [r for r in off if any(r in kids.get(p, []) for p in onpath)]
        r = rng.choice(sib or off)
        opts = list(kids[r])
        rng.shuffle(opts)
        out.append(T.router_example(r, opts, c["state"], "none", c["id"]))
        del path
    return out


_SUB = {}


def subtree(kids, n):
    if n not in _SUB:
        s = set()
        for k in kids.get(n, []):
            s |= subtree(kids, k) if k in kids else {k}
        _SUB[n] = s
    return _SUB[n]


def body_model():
    """Full fine-tune of every layer with the token table frozen: the 'whole body' adapter (~4 MB at the
    shipped int8, against 0.49 MB for a LoRA spike)."""
    m = new_model("full")
    m.enc.embeddings.tok_embeddings.weight.requires_grad_(False)
    return m


def body_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "tok_embeddings" not in k}


def fit(name, exs, seed, smoke, kind="lora"):
    path = OUT / "adapters" / f"{name}.pt"
    model = new_model("lora") if kind == "lora" else body_model()
    if path.exists():
        st = torch.load(path)
        load_adapter(model, st) if kind == "lora" else model.load_state_dict(st, strict=False)
        return model, json.loads((OUT / f"{name}.train.json").read_text())
    rng = random.Random(seed)
    rng.shuffle(exs)
    nval = max(20, len(exs) // 20)
    va, tr = exs[:nval], exs[nval:]
    torch.manual_seed(seed)
    t0 = time.time()
    info = train(model, kind, tr, 1 if smoke else EPOCHS, seed, va=va, max_steps=4 if smoke else 0)
    info.update({"n_examples": len(tr), "train_tokens": int(sum(len(e["ids"]) for e in tr)) * (1 if smoke else EPOCHS),
                 "sec": time.time() - t0})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state(model) if kind == "lora" else body_state(model), path)
    (OUT / f"{name}.train.json").write_text(json.dumps(info, indent=1))
    print(f"[{name}] trained {len(tr)} ex, {info['sec']:.0f}s, best epoch {info['best_epoch']}", flush=True)
    return model, info


def run_queries(model, tok, queries, pad):
    """queries: list of (state, node, options). Returns list of (names, probs, n_tokens)."""
    exs = tok_examples(tok, [T.router_example(n, o, s, None) for s, n, o in queries])
    probs = probs_at(predict(model, exs, pad), 1.0)
    return [(e["names"], [round(x, 5) for x in p], len(e["ids"])) for e, p in zip(exs, probs)]


def tree_calls(model, tok, kids, cases, pad):
    """Every router call of a full expansion of the tree for each case: (case, node) -> probs.
    Full expansion lets analyze.py run greedy and any beam width without the GPU."""
    q, keys = [], []
    for c in cases:
        for node, ch in kids.items():
            q.append((c["state"], node, ch))
            keys.append((c["id"], node))
    res = run_queries(model, tok, q, pad)
    out = {}
    for (cid, node), r in zip(keys, res):
        out.setdefault(cid, {})[node] = r
    return out


def flat_calls(model, tok, leaves, cases, pad):
    long = run_queries(model, tok, [(c["state"], "flat", leaves) for c in cases], pad)
    chunks = [leaves[i:i + SHARD] for i in range(0, len(leaves), SHARD)]
    shard = [run_queries(model, tok, [(c["state"], "flat", ch) for c in cases], pad) for ch in chunks]
    # raw scores are compared across chunks through log-probs relative to each chunk's own `none`
    return {c["id"]: {"long": long[i], "shards": [s[i] for s in shard]} for i, c in enumerate(cases)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = ROOT / "out" / "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    from lora import stock_path
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    pad = tok.tok.token_to_id("[PAD]")
    import lora
    lora.PAD = pad
    lc = T.leaf_cases()
    train_cases = [c for x in T.TRAINED for c in lc[x]["train"]]
    if a.smoke:
        train_cases = random.Random(0).sample(train_cases, 40)
    rng = random.Random("hier-router")
    flat_ex = tok_examples(tok, flat_train(train_cases, rng))
    tree_ex = tok_examples(tok, tree_train(train_cases, rng))
    flat, finfo = fit("flat", flat_ex, 0, a.smoke)
    treem, tinfo = fit("tree", tree_ex, 0, a.smoke)
    bodym, binfo = fit("tree_body", tree_ex, 0, a.smoke, kind="full")

    ev_path = OUT / "eval.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {"configs": {}}
    ev["train"] = {"flat": finfo, "tree": tinfo, "tree_body": binfo}
    lim = 3 if a.smoke else None
    configs = []
    for n in SIZES:
        for d in range(1 if n == len(T.TRAINED) else DRAWS):
            leaves = sorted(random.Random(f"draw-{n}-{d}").sample(T.TRAINED, n), key=T.LEAVES.index)
            configs.append((f"N{n}-d{d}", leaves, leaves))
    configs.append(("absent", T.TRAINED, T.LEAVES))          # held-out leaves not in the library
    configs.append(("added", T.LEAVES, T.LEAVES))            # held-out leaves added by description only
    for name, lib, test_leaves in configs:
        if name in ev["configs"]:
            continue
        t0 = time.time()
        cases = [c for x in test_leaves for c in lc[x]["test"][:lim]]
        kids = T.induced(lib)
        _SUB.clear()
        ev["configs"][name] = {
            "library": lib, "kids": kids, "cases": [[c["id"], c["leaf"]] for c in cases],
            "tree": tree_calls(treem, tok, kids, cases, pad),
            "tree_body": tree_calls(bodym, tok, kids, cases, pad),
            "flat": flat_calls(flat, tok, lib, cases, pad),
            "sec": time.time() - t0,
        }
        ev_path.write_text(json.dumps(ev))
        print(f"[{name}] {len(lib)} leaves, {len(cases)} cases, {time.time() - t0:.0f}s", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
