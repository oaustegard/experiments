"""Teacher-forced metrics for looped Monad arms on the pinned SYNTH set (data/eval_synth.jsonl).

For every row: prompt P, reasoning trace R, separator S = "\\n</think>\\n\\n", answer A + "<|im_end|>".
  ans_nll@f   mean NLL per answer token with only the first f of the trace kept (f = 0, 0.5, 1).
              If loops move reasoning into the latent, ans_nll@0 and @0.5 should fall more than @1.
  trace_nll   mean NLL per trace token (f = 1): plain language-model quality on reasoning text.
  kl          mean KL(base || arm) over trace and answer positions (f = 1).
Token-weighted means per group (memorization, math, mcq) and overall.

    python3 tf_eval.py --configs configs.json --split test --out results/tf_screen.json [--model DIR]

configs.json: [{"name": "base"}, {"name": "B1_K2_a0.25", "band": [12, 20], "K": 2, "alpha": 0.25}, ...]
An entry without "alpha" uses the checkpoint's learned gate.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loopmodel import LoopedMonad  # noqa: E402

FRACS = (0.0, 0.5, 1.0)


def encode_rows(tok, rows):
    out = []
    for r in rows:
        p = tok.encode(f"<|im_start|>user\n{r['q']}<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False)
        t = tok.encode(r["reasoning"], add_special_tokens=False)
        a = tok.encode(f"\n</think>\n\n{r['answer']}<|im_end|>", add_special_tokens=False)
        out.append({"group": r["group"], "p": p, "t": t, "a": a})
    return out


def build(e, f):
    """ids, and position-class labels: 0 prompt, 1 trace, 2 answer (separator counts as answer)."""
    kt = int(len(e["t"]) * f)
    ids = e["p"] + e["t"][:kt] + e["a"]
    cls = [0] * len(e["p"]) + [1] * kt + [2] * len(e["a"])
    return ids, cls


def batches(items, max_tokens):
    items = sorted(items, key=lambda x: len(x[1]))
    b, L = [], 0
    for it in items:
        L2 = max(L, len(it[1]))
        if b and L2 * (len(b) + 1) > max_tokens:
            yield b
            b, L2 = [], len(it[1])
        b.append(it)
        L = L2
    if b:
        yield b


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--model", default=str(HERE / "models/monad"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="rows per group (smoke runs)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp32")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(HERE / "data/eval_synth.jsonl")]
    rows = [r for r in rows if r["split"] == a.split]
    if a.limit:
        keep = {}
        rows = [r for r in rows if keep.setdefault(r["group"], []).append(1) or len(keep[r["group"]]) <= a.limit]
    enc = encode_rows(tok, rows)
    configs = json.loads(Path(a.configs).read_text())
    bands = {tuple(c["band"]) for c in configs if c.get("band")}
    kmax = max([c.get("K", 1) for c in configs] + [2])
    # one model object; band is switched per config (weights are identical, only the schedule changes)
    model = LoopedMonad.from_pretrained(a.model, band=None, kmax=kmax).to(dev, dtype).eval()
    learned = (model.band, model.gate.detach().clone()) if model.band else None
    groups = sorted({e["group"] for e in enc})
    acc = {c["name"]: {g: {"ans": [0.0] * len(FRACS), "ans_n": [0] * len(FRACS), "trace": 0.0, "trace_n": 0,
                           "kl": 0.0, "kl_n": 0} for g in groups} for c in configs}
    t0 = time.time()
    for fi, f in enumerate(FRACS):
        items = [(e, *build(e, f)) for e in enc]
        for bi, b in enumerate(batches(items, a.max_tokens)):
            L = max(len(x[1]) for x in b)
            ids = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long)
            cls = torch.full((len(b), L), -1, dtype=torch.long)
            for r, (_, i, c) in enumerate(b):
                ids[r, :len(i)] = torch.tensor(i)
                cls[r, :len(c)] = torch.tensor(c)
            ids, cls = ids.to(dev), cls.to(dev)
            tgt, tcls = ids[:, 1:], cls[:, 1:]
            base_lp = None
            for c in configs:
                band = tuple(c["band"]) if c.get("band") else None
                K = c.get("K", 1)
                if band and "alpha" not in c:
                    assert learned and tuple(learned[0]) == band, f"{c['name']}: no learned gate for band {band}"
                model.band = band
                logits = model(ids, K=K, alpha=c.get("alpha"))[:, :-1].float()
                lp = F.log_softmax(logits, -1)
                nll = -lp.gather(-1, tgt[..., None])[..., 0]
                if c["name"] == configs[0]["name"] and f == 1.0:
                    base_lp = lp
                for r, (e, _, _) in enumerate(b):
                    g = acc[c["name"]][e["group"]]
                    am = tcls[r] == 2
                    g["ans"][fi] += nll[r][am].sum().item()
                    g["ans_n"][fi] += int(am.sum())
                    if f == 1.0:
                        tm = tcls[r] == 1
                        g["trace"] += nll[r][tm].sum().item()
                        g["trace_n"] += int(tm.sum())
                        if base_lp is not None and c["name"] != configs[0]["name"]:
                            km = tcls[r] >= 1
                            kl = (base_lp[r][km].exp() * (base_lp[r][km] - lp[r][km])).sum(-1)
                            g["kl"] += kl.sum().item()
                            g["kl_n"] += int(km.sum())
            print(f"frac {f} batch {bi} rows {len(b)} len {L} {time.time() - t0:.0f}s", flush=True)
    res = {}
    for name, gs in acc.items():
        def summ(sel):
            d = {}
            for fi, f in enumerate(FRACS):
                n = sum(gs[g]["ans_n"][fi] for g in sel)
                d[f"ans_nll@{f}"] = round(sum(gs[g]["ans"][fi] for g in sel) / max(n, 1), 4)
            n = sum(gs[g]["trace_n"] for g in sel)
            d["trace_nll"] = round(sum(gs[g]["trace"] for g in sel) / max(n, 1), 4)
            n = sum(gs[g]["kl_n"] for g in sel)
            d["kl"] = round(sum(gs[g]["kl"] for g in sel) / n, 4) if n else 0.0
            return d
        res[name] = {"all": summ(groups), **{g: summ([g]) for g in groups}}
    cfg_by = {c["name"]: c for c in configs}
    out = {"split": a.split, "rows": len(enc), "model": a.model, "device": dev, "dtype": a.dtype,
           "elapsed_s": round(time.time() - t0, 1), "configs": cfg_by, "results": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name in res:
        r = res[name]["all"]
        print(f"{name:24s} " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    main()
