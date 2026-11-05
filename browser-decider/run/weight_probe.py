"""Inspect the shipped 10.2 MB decider for size and accuracy levers, from the packed file alone.

    python3 -I run/weight_probe.py [--sub N] [--skip-full]

Reads web/models/decider/{manifest.json,weights.bin,tokenizer.json} (the browser build), runs the
NumPy reference forward pass (pack.Ref) on the Typed Decisions test split and writes
results/weight_probe.json:
  bytes      where the file's bytes go
  vocab      token ids the benchmark's text actually uses (train+test+decider_eval), and the table
             bytes a pruned vocabulary would take
  full       the shipped file's test score (must reproduce 0.730)
  ablate     each layer / attention block / MLP block / head removed: agreement with the unablated
             model's top option and mean KL from its distribution, on the first N test questions
  acts       residual-stream outliers per layer (max |x| per channel against the median channel)
  quant      int8 code use per matrix (share of columns whose codes span the full +-63)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402
from pack import Ref, gelu  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

MD = ROOT / "web" / "models" / "decider"
MAX_LEN = 1024


class Enc:
    """train_enc.Enc without its torch/transformers imports (same token layout)."""

    def __init__(self, tok_path):
        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.cls, self.sep, self.mask = (self.tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[MASK]"))

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def build(self, e):
        ids = [self.cls] + self.ids(e["question"]) + [self.sep]
        pos = []
        for name, desc in e["options"]:
            pos.append(len(ids))
            ids += [self.mask] + self.ids(f" {name}: {desc}")
        ids.append(self.sep)
        budget = MAX_LEN - len(ids) - 1
        ids += self.ids(e["state"])[:budget] + [self.sep]
        return ids, pos


class Probe(Ref):
    """Ref with switches: drop a layer, its attention, its MLP, or one head; record activations."""

    def __init__(self, m, data):
        super().__init__(m, data)
        self.off = {}      # (layer, "attn"|"mlp"|"all") -> True, or (layer, "head", h)
        self.rec = None    # list per layer of residual [S, H] after the layer

    def hidden(self, ids):
        m = self.m
        H, nh = m["hidden"], m["heads"]
        D = H // nh
        S = len(ids)
        x = self.ln(self.table(ids), self.t(m["emb_norm"]))
        pos = np.arange(S)
        local = np.abs(pos[:, None] - pos[None]) <= m["window"] // 2
        for li, L in enumerate(m["layers"]):
            if self.off.get((li, "all")):
                if self.rec is not None:
                    self.rec.append(x)
                continue
            if not self.off.get((li, "attn")):
                h = x if L["attn_norm"] is None else self.ln(x, self.t(L["attn_norm"]))
                qkv = self.lin(h, L["Wqkv"]).reshape(S, 3, nh, D).transpose(1, 2, 0, 3)
                th = m["theta_global"] if L["global"] else m["theta_local"]
                q, k, v = self.rope(qkv[0], th), self.rope(qkv[1], th), qkv[2]
                sc = np.einsum("hid,hjd->hij", q, k) / math.sqrt(D)
                if not L["global"]:
                    sc = np.where(local, sc, -np.inf)
                sc = np.exp(sc - sc.max(-1, keepdims=True))
                sc /= sc.sum(-1, keepdims=True)
                o = np.einsum("hij,hjd->hid", sc, v)
                for hh in range(nh):
                    if self.off.get((li, "head", hh)):
                        o[hh] = 0
                o = o.transpose(1, 0, 2).reshape(S, H)
                x = x + self.lin(o, L["Wo"])
            if not self.off.get((li, "mlp")):
                u = self.lin(self.ln(x, self.t(L["mlp_norm"])), L["Wi"])
                I = m["inter"]
                x = x + self.lin(gelu(u[:, :I]) * u[:, I:], L["Wo_mlp"])
            if self.rec is not None:
                self.rec.append(x)
        return self.ln(x, self.t(m["final_norm"]))


def probs_of(p, exs, temps):
    out = []
    for e in exs:
        z = p.logits(e["ids"], e["pos"]) / temps[e["type"]]
        z = np.exp(z - z.max())
        out.append(z / z.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=400, help="test questions used for ablations")
    ap.add_argument("--skip-full", action="store_true")
    a = ap.parse_args()
    man = json.loads((MD / "manifest.json").read_text())
    data = (MD / "weights.bin").read_bytes()
    p = Probe(man, data)
    tok = Enc(MD / "tokenizer.json")
    res = {}

    # bytes
    T = man["table"]
    tb = T["codes"]["length"] + T["scales"]["length"]
    lay = sum(r["length"] for L in man["layers"] for w in L.values() if isinstance(w, dict)
              for r in ([w] if "offset" in w else [v for v in w.values() if isinstance(v, dict)]))
    res["bytes"] = {"file": len(data), "token_table": tb, "layers": lay, "per_token_row": tb / T["V"]}

    # vocab use
    W = WORKFLOWS
    tr = examples([c for w in W for c in load_cases(w, "train")])
    te = examples([c for w in W for c in load_cases(w, "test")])
    for e in tr + te:
        e["ids"], e["pos"] = tok.build(e)
    used_tr = {i for e in tr for i in e["ids"]}
    used_te = {i for e in te for i in e["ids"]}
    ood = [json.loads(l) for l in (ROOT / "data" / "decider_eval.jsonl").read_text().splitlines() if l.strip()]
    used_ood = set()
    for r in ood:
        used_ood |= set(tok.ids(json.dumps(r)))
    unseen_te = used_te - used_tr
    res["vocab"] = {"V": T["V"], "train": len(used_tr), "test": len(used_te), "train_or_test": len(used_tr | used_te),
                    "test_not_in_train": len(unseen_te),
                    "test_tokens_not_in_train": sum(1 for e in te for i in e["ids"] if i in unseen_te),
                    "test_tokens": sum(len(e["ids"]) for e in te),
                    "decider_eval": len(used_ood), "decider_eval_not_in_train": len(used_ood - used_tr)}
    print("vocab", res["vocab"], flush=True)

    # full test score of the shipped file
    if not a.skip_full:
        t0 = time.time()
        base_full = probs_of(p, te, man["temps"])
        summ, _ = score_examples(te, [b.tolist() for b in base_full])
        res["full"] = {**summ, "sec": round(time.time() - t0, 1)}
        print("full", res["full"], flush=True)

    # ablations on a subset, measured against the unablated model
    sub = te[: a.sub]
    base = probs_of(p, sub, man["temps"])
    gold_acc = lambda P: float(np.mean([int(np.argmax(q)) == int(np.argmax(e["gold"])) for q, e in zip(P, sub)]))
    res["ablate_base_acc"] = gold_acc(base)

    def measure(key):
        p.off = {key: True}
        P = probs_of(p, sub, man["temps"])
        p.off = {}
        agree = float(np.mean([int(np.argmax(x)) == int(np.argmax(y)) for x, y in zip(P, base)]))
        kl = float(np.mean([np.sum(y * np.log(np.maximum(y, 1e-12) / np.maximum(x, 1e-12))) for x, y in zip(P, base)]))
        return {"agree": round(agree, 4), "kl": round(kl, 4), "acc": round(gold_acc(P), 4)}

    res["ablate"] = {}
    nL, nh = len(man["layers"]), man["heads"]
    for li in range(nL):
        for part in ("all", "attn", "mlp"):
            k = f"L{li}.{part}"
            res["ablate"][k] = measure((li, part))
            print(k, res["ablate"][k], flush=True)
        for hh in range(nh):
            k = f"L{li}.head{hh}"
            res["ablate"][k] = measure((li, "head", hh))
            print(k, res["ablate"][k], flush=True)

    # activation outliers on 40 questions
    p.rec = []
    stats = [[] for _ in range(nL)]
    for e in sub[:40]:
        p.rec = []
        p.hidden(np.array(e["ids"]))
        for li, x in enumerate(p.rec):
            stats[li].append(np.abs(x).max(0))
    p.rec = None
    acts = []
    for li in range(nL):
        mx = np.max(stats[li], 0)
        order = np.argsort(-mx)
        acts.append({"layer": li, "max": round(float(mx[order[0]]), 2), "median_channel_max": round(float(np.median(mx)), 2),
                     "ratio": round(float(mx[order[0]] / np.median(mx)), 1), "top_channels": [int(c) for c in order[:3]]})
    res["acts"] = acts
    print("acts", acts, flush=True)

    # int8 code use
    q = {}
    for li, L in enumerate(man["layers"]):
        for name in ("Wqkv", "Wo", "Wi", "Wo_mlp"):
            c = p.t(L[name]["q"], np.int8).astype(np.int16)
            rms = np.sqrt((c.astype(np.float32) ** 2).mean(0))
            q[f"L{li}.{name}"] = {"col_rms_codes_median": round(float(np.median(rms)), 1),
                                  "col_max_over_rms_median": round(float(np.median(63 / rms)), 1)}
    res["quant"] = q
    (ROOT / "results" / "weight_probe.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote results/weight_probe.json")


if __name__ == "__main__":
    main()
