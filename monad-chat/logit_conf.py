"""Does Monad's own token probability tell right answers from wrong ones? Loop-fixed 8-bit, greedy,
onnxruntime CPU, on the 30 fact questions and the 40 yes/no questions. For each answer: the probability
the model gave each token it wrote, summarised over the answer (after </think>) and over the trace.
AUROC of each summary against correctness; 0.5 = no signal.
    python3 logit_conf.py -> runs/logit_conf.jsonl
"""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monad-weight-edit"))
from ort_gen import session, generate, pieces  # noqa: E402
from bridge import tok  # noqa: E402
from yesno_qs import YES, NO  # noqa: E402

t = tok(); P = pieces(t)
s = session(str(HERE.parent / "monad-weight-edit/export/A8p_q8"), threads=4)
facts = json.loads((HERE / "probes_facts.json").read_text())
items = [("fact", p["q"], p["check"]) for p in facts] + [("yesno", q, "yes") for q in YES] + [("yesno", q, "no") for q in NO]


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


rows = []
for k in range(0, len(items), 10):
    chunk = items[k:k + 10]
    r = generate(s, [t.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False) for _, q, _ in chunk], 768, P, record_logits=769)
    L = np.stack(r["logits"])                      # steps, B, V
    for b, (kind, q, gold) in enumerate(chunk):
        toks = r["tokens"][b]
        lg = L[:len(toks), b].astype(np.float64)
        lp = lg - lg.max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        p_tok = np.exp(lp[np.arange(len(toks)), toks])
        ent = -(np.exp(lp) * lp).sum(-1)
        text, cut = "", None
        for i, tk in enumerate(toks):
            text += P[tk]
            if cut is None and "</think>" in text:
                cut = i + 1
        ans = text.split("</think>", 1)[1].split("<|im_end|>")[0].strip() if cut else ""
        a = slice(cut, None) if cut else slice(len(toks), None)
        pa, ea = p_tok[a][:60], ent[a][:60]
        while len(pa) and P[toks[cut]].strip() == "":        # skip leading whitespace tokens
            pa, ea, cut = pa[1:], ea[1:], cut + 1
        if kind == "fact":
            ok = bool(re.search(gold, ans, re.I))
        else:
            m = re.match(r"\W*(yes|no)\b", ans, re.I); ok = bool(m and m.group(1).lower() == gold)
        rows.append({"kind": kind, "q": q, "ok": ok, "answer": ans[:200],
                     "ans_mean_p": float(pa.mean()) if len(pa) else None, "ans_min_p20": float(pa[:20].min()) if len(pa) else None,
                     "ans_first_p": float(pa[0]) if len(pa) else None, "ans_mean_ent": float(ea.mean()) if len(ea) else None,
                     "think_mean_p": float(p_tok[:cut].mean()) if cut else float(p_tok.mean())})
    print(k, flush=True)
(HERE / "runs/logit_conf.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
for kind in ("fact", "yesno", "all"):
    rs = [x for x in rows if (kind == "all" or x["kind"] == kind) and x["ans_mean_p"] is not None]
    y = [x["ok"] for x in rs]
    print(kind, f"n={len(rs)} right={sum(y)}",
          {f: round(auroc([x[f] if f != "ans_mean_ent" else -x[f] for x in rs], y) or 0, 3)
           for f in ("ans_mean_p", "ans_min_p20", "ans_first_p", "ans_mean_ent", "think_mean_p")})
