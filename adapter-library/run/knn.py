"""CPU arms: embed every case's state with the 10 MB build's embedder, then routing, gap detection and
the kNN growth curve.

    python3 -I run/knn.py

Writes out/emb.npz (cache), results/routing.json, results/knn.json and out/knn_probs.json (per-test-
decision kNN distributions for every (W, seed, n), which analyze.py mixes with the GPU arms).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import E17C, KNN_SEEDS, ROOT, SIZES, WORKFLOWS, examples, load_cases, revealed, score_examples  # noqa: E402

K, TAU, EPS = 8, 0.05, 0.01
OUT, RES = ROOT / "out", ROOT / "results"


@torch.no_grad()
def embed(texts, bs=16):
    # tokenizer_config.json names transformers 5's TokenizersBackend; load the tokenizer.json directly
    tok = PreTrainedTokenizerFast(tokenizer_file=str(E17C / "tokenizer.json"), pad_token="[PAD]")
    m = AutoModel.from_pretrained(str(E17C)).eval()
    dense = load_file(str(E17C / "2_Dense/model.safetensors"))["linear.weight"]
    torch.set_num_threads(4)
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=1024, return_tensors="pt", return_token_type_ids=False)
        h = m(**b).last_hidden_state
        a = b["attention_mask"].unsqueeze(-1).float()
        v = ((h * a).sum(1) / a.sum(1)) @ dense.T
        out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        print(f"embed {i + len(b['input_ids'])}/{len(texts)}", flush=True)
    return np.concatenate(out)


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    OUT.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    cases = {(w, s): load_cases(w, s) for w in WORKFLOWS for s in ("train", "test")}
    ids = [c["id"] for v in cases.values() for c in v]
    cache = OUT / "emb.npz"
    if cache.exists() and list(np.load(cache)["ids"]) == ids:
        X = np.load(cache)["X"]
    else:
        X = embed([c["state"] for v in cases.values() for c in v])
        np.savez(cache, ids=np.array(ids), X=X)
    E = dict(zip(ids, X))

    # --- R: routing over all four, gap detection with W left out ---
    tr = [(c["id"], w) for w in WORKFLOWS for c in cases[(w, "train")]]
    TrX, TrW = np.stack([E[i] for i, _ in tr]), np.array([w for _, w in tr])
    route_ok, per_w = 0, {}
    test_all = [(c["id"], w) for w in WORKFLOWS for c in cases[(w, "test")]]
    for cid, w in test_all:
        top = np.argsort(-(TrX @ E[cid]))[:K]
        vals, cnt = np.unique(TrW[top], return_counts=True)
        ok = vals[cnt.argmax()] == w
        route_ok += ok
        per_w.setdefault(w, []).append(ok)
    gap = {}
    for w in WORKFLOWS:
        keep = TrW != w
        S = lambda cid: 1 - float((TrX[keep] @ E[cid]).max())
        pos = np.array([S(cid) for cid, ww in test_all if ww == w])
        neg = np.array([S(cid) for cid, ww in test_all if ww != w])
        # routing W's cases when W is absent: which library spike would they go to
        lib = {}
        for cid, ww in test_all:
            if ww != w:
                continue
            top = np.argsort(-(TrX[keep] @ E[cid]))[:K]
            vals, cnt = np.unique(TrW[keep][top], return_counts=True)
            lib[cid] = str(vals[cnt.argmax()])
        gap[w] = {"auroc": auroc(pos, neg), "gap_score_mean_W": float(pos.mean()),
                  "gap_score_mean_others": float(neg.mean()), "routed_to": lib}
    routing = {"k": K, "accuracy": route_ok / len(test_all),
               "by_workflow": {w: float(np.mean(v)) for w, v in per_w.items()},
               "gap": {w: {k: v for k, v in g.items() if k != "routed_to"} for w, g in gap.items()}}
    (RES / "routing.json").write_text(json.dumps(routing, indent=1))
    (OUT / "routed_to.json").write_text(json.dumps({w: g["routed_to"] for w, g in gap.items()}))
    print(json.dumps(routing, indent=1))

    # --- G/knn: growth curve on each W from its own revealed cases ---
    knn, probs_out = {}, {}
    for w in WORKFLOWS:
        te = examples(cases[(w, "test")])
        for seed in KNN_SEEDS:
            for n in SIZES:
                key = f"{w}|s{seed}|n{n}"
                if n == 0:
                    probs = [[1 / len(e["options"])] * len(e["options"]) for e in te]
                else:
                    rev = examples(revealed(w, seed, n))
                    by_q = {}
                    for e in rev:
                        by_q.setdefault(e["qname"], []).append((E[e["case"]], np.array(e["gold"])))
                    probs = []
                    for e in te:
                        keys = np.stack([k for k, _ in by_q[e["qname"]]])
                        golds = np.stack([g for _, g in by_q[e["qname"]]])
                        s = keys @ E[e["case"]]
                        top = np.argsort(-s)[: min(K, len(s))]
                        wt = np.exp((s[top] - s[top].max()) / TAU)
                        p = (wt[:, None] * golds[top]).sum(0) / wt.sum()
                        p = (1 - EPS) * p + EPS / len(p)
                        probs.append(p.tolist())
                summ, _ = score_examples(te, probs)
                knn[key] = summ
                probs_out[key] = [[round(x, 5) for x in p] for p in probs]
                print(f"knn {key}: acc {summ['accuracy']:.4f} kl {summ['kl_from_gold']:.4f}", flush=True)
    (RES / "knn.json").write_text(json.dumps({"k": K, "tau": TAU, "eps": EPS, "runs": knn}, indent=1))
    (OUT / "knn_probs.json").write_text(json.dumps(probs_out))


if __name__ == "__main__":
    main()
