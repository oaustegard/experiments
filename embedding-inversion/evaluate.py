"""Generate hypotheses and run the correction loop.

  python3 evaluate.py gen  --cond float --split train   # zero-step hyps for corrector training
  python3 evaluate.py eval --cond float --rounds 5       # the numbers in RESULTS.md

`gen` writes data/hyps_<cond>_<split>.json: greedy zero-step hypothesis per
item plus its bekko embedding (float; the condition is applied at load).

`eval` reports, per round, exact match / token F1 / BLEU / verifier cosine,
plus two controls: nearest training string by cosine (memorization floor) and
the zero-step top beam without the verifier picking among beams. Results go to
results_<cond>.json with every per-item hypothesis kept for recheck.py.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sacrebleu
import torch

from encoder import BekkoEncoder, SignBits, condition
from model import Inverter, encode_labels, tokenizer

HERE = Path(__file__).resolve().parent
DATA, CKPT = HERE / "data", HERE / "ckpt"


def norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").split())


def token_f1(a: str, b: str) -> float:
    ta, tb = norm(a).split(), norm(b).split()
    common = sum((Counter(ta) & Counter(tb)).values())
    if not ta or not tb or common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def metrics(refs, hyps, cos):
    em = float(np.mean([norm(r) == norm(h) for r, h in zip(refs, hyps)]))
    f1 = float(np.mean([token_f1(r, h) for r, h in zip(refs, hyps)]))
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return {"exact": em, "token_f1": f1, "bleu": bleu, "cosine": float(np.mean(cos))}


def load(mode, cond, k):
    m = Inverter(mode, k=k)
    m.load_state_dict(torch.load(CKPT / f"{mode}_{cond}.pt", map_location="cpu"))
    m.eval()
    return m


def batched(n, bs):
    for s in range(0, n, bs):
        yield slice(s, s + bs)


def generate_zero(model, tok, emb, bs, beams, nret):
    out = []
    for sl in batched(len(emb), bs):
        ids = model.generate(emb[sl], num_beams=beams, num_return_sequences=nret)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def generate_correct(model, tok, emb, hyps, hyp_emb, bs, beams, nret):
    out = []
    for sl in batched(len(emb), bs):
        _, hid, hm = encode_labels(tok, hyps[sl])
        ids = model.generate(emb[sl], hyp_emb[sl], hid, hm, num_beams=beams, num_return_sequences=nret)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "eval"])
    ap.add_argument("--cond", choices=["float", "bin1"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    sb = SignBits.load(DATA / "signbits_mu.npy") if a.cond == "bin1" else None
    texts = json.loads((DATA / "splits.json").read_text())[a.split]
    emb_f = np.load(DATA / f"emb_{a.split}.npy")
    if a.n:
        texts, emb_f = texts[: a.n], emb_f[: a.n]
    emb = torch.from_numpy(condition(a.cond, emb_f, sb))
    tok = tokenizer()
    enc = BekkoEncoder()
    see = lambda x: condition(a.cond, x, sb)  # what the verifier sees

    if a.cmd == "gen":
        model = load("zero", a.cond, a.k)
        hyps = generate_zero(model, tok, emb, a.bs, 1, 1)
        he = enc.encode(hyps, batch_size=64)
        Path(DATA / f"hyps_{a.cond}_{a.split}.json").write_text(
            json.dumps({"hyp": hyps, "emb": he.round(6).tolist()}))
        cos = (see(he) * see(emb_f)).sum(1)
        print(f"gen {a.cond}/{a.split}: n={len(hyps)} greedy zero-step "
              f"{json.dumps(metrics(texts, hyps, cos))} ({time.time()-t0:.0f}s)", flush=True)
        return

    res = {"cond": a.cond, "n": len(texts), "rounds": {}, "items": []}
    tgt = see(emb_f)

    # control 1: nearest training string under the same verifier
    tr_texts = json.loads((DATA / "splits.json").read_text())["train"]
    tr_emb = see(np.load(DATA / "emb_train.npy"))
    nn_idx = (tgt @ tr_emb.T).argmax(1)
    nn_hyp = [tr_texts[i] for i in nn_idx]
    nn_cos = (tgt * tr_emb[nn_idx]).sum(1)
    res["rounds"]["nn_train"] = metrics(texts, nn_hyp, nn_cos)
    print("nn_train", res["rounds"]["nn_train"], flush=True)

    # round 0: zero-step, beam search; keep top beam AND verifier-best beam
    zero = load("zero", a.cond, a.k)
    cands = generate_zero(zero, tok, emb, a.bs, a.beams, a.beams)
    cands = [cands[i * a.beams:(i + 1) * a.beams] for i in range(len(texts))]
    flat = [c for cs in cands for c in cs]
    ce = see(enc.encode(flat, batch_size=64)).reshape(len(texts), a.beams, -1)
    cos = np.einsum("nbd,nd->nb", ce, tgt)
    top = [cs[0] for cs in cands]
    res["rounds"]["zero_top_beam"] = metrics(texts, top, cos[:, 0])
    best = cos.argmax(1)
    hyp = [cs[b] for cs, b in zip(cands, best)]
    hyp_e = ce[np.arange(len(texts)), best]
    hyp_cos = cos[np.arange(len(texts)), best]
    res["rounds"]["round0"] = metrics(texts, hyp, hyp_cos)
    print("zero_top_beam", res["rounds"]["zero_top_beam"], flush=True)
    print("round0", res["rounds"]["round0"], f"({time.time()-t0:.0f}s)", flush=True)
    history = [list(hyp)]

    if (CKPT / f"correct_{a.cond}.pt").exists() and a.rounds > 0:
        corr = load("correct", a.cond, a.k)
        del zero
        for r in range(1, a.rounds + 1):
            cands = generate_correct(corr, tok, emb, hyp, torch.from_numpy(hyp_e), a.bs, a.beams, a.beams)
            cands = [cands[i * a.beams:(i + 1) * a.beams] for i in range(len(texts))]
            flat = [c for cs in cands for c in cs]
            ce = see(enc.encode(flat, batch_size=64)).reshape(len(texts), a.beams, -1)
            cos = np.einsum("nbd,nd->nb", ce, tgt)
            # keep the incumbent unless a candidate re-embeds closer
            best = cos.argmax(1)
            improved = cos[np.arange(len(texts)), best] > hyp_cos
            new_hyp = [cs[b] if imp else h for cs, b, imp, h in zip(cands, best, improved, hyp)]
            hyp_e = np.where(improved[:, None], ce[np.arange(len(texts)), best], hyp_e)
            hyp_cos = np.where(improved, cos[np.arange(len(texts)), best], hyp_cos)
            hyp = new_hyp
            history.append(list(hyp))
            res["rounds"][f"round{r}"] = metrics(texts, hyp, hyp_cos)
            res["rounds"][f"round{r}"]["improved_frac"] = float(improved.mean())
            print(f"round{r}", res["rounds"][f"round{r}"], f"({time.time()-t0:.0f}s)", flush=True)
    else:
        res["note"] = "no corrector checkpoint; rounds beyond 0 skipped"

    # length curve on the final hypothesis
    wl = np.array([len(t.split()) for t in texts])
    final = hyp
    buckets = [(0, 6), (7, 10), (11, 16), (17, 999)]
    res["by_length"] = {}
    for lo, hi in buckets:
        m = (wl >= lo) & (wl <= hi)
        if m.sum():
            res["by_length"][f"{lo}-{hi if hi < 999 else 'plus'}"] = {
                "n": int(m.sum()),
                "exact": float(np.mean([norm(texts[i]) == norm(final[i]) for i in np.where(m)[0]])),
                "token_f1": float(np.mean([token_f1(texts[i], final[i]) for i in np.where(m)[0]])),
                "cosine": float(hyp_cos[m].mean()),
            }
    res["items"] = [{"text": t, "words": int(w), "nn": nn_hyp[i], "hyps": [h[i] for h in history],
                     "final_cos": float(hyp_cos[i])} for i, (t, w) in enumerate(zip(texts, wl))]
    res["secs"] = time.time() - t0
    (HERE / f"results_{a.cond}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
