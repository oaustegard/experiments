"""Fixed-schema text classifier arms for the msd-context-classifier experiment.

Arms:
  --arm ft     fine-tune an encoder (mean-pool + linear head), CPU
  --arm probe  frozen encoder mean-pool embeddings + logistic regression (the cheap floor)

Data: data/corpus.jsonl rows {"url","label","title","text"}; split by url hash (stratified).
Optional --queries data/queries.jsonl rows {"text","label"} evaluated with the trained model
(the transfer test: page-trained model on question-shaped inputs).

Metrics: accuracy, macro-F1, top-label ECE (15 bins) raw and after dev-fit temperature,
coverage at a 5% error budget (threshold chosen on dev, applied to test), ms/example batch 1.
Restartable: --arm ft checkpoints every epoch to models/ckpt_<name>_ep<k>.pt and resumes.
Completion line: 'done -> <out json>'.
"""
import argparse, json, os, sys, time, math, hashlib, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_corpus(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def split_rows(rows, seed=20260920, frac=(0.70, 0.15, 0.15)):
    """Stratified by label, deterministic by url hash + seed."""
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    tr, dv, te = [], [], []
    for lab, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: hashlib.sha1((str(seed) + r["url"]).encode()).hexdigest())
        n = len(rs); a = int(n * frac[0]); b = a + int(n * frac[1])
        tr += rs[:a]; dv += rs[a:b]; te += rs[b:]
    rng = random.Random(seed); rng.shuffle(tr)
    return tr, dv, te


def texts_of(rows):
    return [((r.get("title") or "").strip() + "\n" + (r.get("text") or "").strip()).strip() for r in rows]


class Clf(nn.Module):
    def __init__(self, enc, n_labels, dropout=0.1):
        super().__init__()
        self.enc = enc
        h = enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, n_labels)

    def pool(self, out, mask):
        m = mask.unsqueeze(-1).float()
        return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.drop(self.pool(out, attention_mask)))


def batches(tok, texts, max_len, bs):
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict_logits(model, tok, texts, max_len, bs):
    model.eval(); outs = []
    for ids, am in batches(tok, texts, max_len, bs):
        outs.append(model(ids, am).float())
    return torch.cat(outs).numpy() if outs else np.zeros((0, 1))


@torch.no_grad()
def embed(enc, tok, texts, max_len, bs):
    enc.eval(); outs = []
    for ids, am in batches(tok, texts, max_len, bs):
        out = enc(input_ids=ids, attention_mask=am)
        m = am.unsqueeze(-1).float()
        e = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)
        outs.append(F.normalize(e, dim=-1))
    return torch.cat(outs).numpy()


# ---------- metrics ----------
def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(tot)


def macro_f1(pred, y, n):
    f = []
    for c in range(n):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def fit_temperature(logits, y):
    """Grid + refine on NLL. Returns T."""
    best = (1e9, 1.0)
    for T in np.concatenate([np.linspace(0.3, 5.0, 48), [1.0]]):
        p = softmax(logits, T); nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best[0]: best = (nll, float(T))
    return best[1]


def coverage_at_budget(dev_probs, dev_y, te_probs, te_y, budget=0.05):
    """Pick the lowest confidence threshold on dev whose selective error <= budget; apply to test."""
    conf_d = dev_probs.max(1); ok_d = dev_probs.argmax(1) == dev_y
    ths = np.unique(np.round(conf_d, 3))[::-1]; chosen = None
    for t in ths:
        m = conf_d >= t
        if m.mean() < 0.02: continue
        if 1 - ok_d[m].mean() <= budget: chosen = float(t)
    if chosen is None:
        return {"threshold": None, "coverage": 0.0, "error_at_coverage": None}
    m = te_probs.max(1) >= chosen; ok_t = te_probs.argmax(1) == te_y
    return {"threshold": chosen, "coverage": float(m.mean()), "error_at_coverage": float(1 - ok_t[m].mean()) if m.any() else None}


def evaluate(name, logits_dev, y_dev, logits_te, y_te, n_labels, labels, extra=None):
    T = fit_temperature(logits_dev, y_dev)
    p_raw = softmax(logits_te); p_cal = softmax(logits_te, T); pred = p_raw.argmax(1)
    res = {
        "n_test": int(len(y_te)), "accuracy": float((pred == y_te).mean()), "macro_f1": macro_f1(pred, y_te, n_labels),
        "ece_raw": ece(p_raw, y_te), "ece_temp": ece(p_cal, y_te), "temperature": T,
        "coverage_5pct": coverage_at_budget(softmax(logits_dev, T), y_dev, p_cal, y_te, 0.05),
        "per_label_f1": {},
    }
    for c in range(n_labels):
        tp = ((pred == c) & (y_te == c)).sum(); fp = ((pred == c) & (y_te != c)).sum(); fn = ((pred != c) & (y_te == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        res["per_label_f1"][labels[c]] = {"f1": float(2 * p * r / (p + r) if p + r else 0.0), "n": int((y_te == c).sum())}
    if extra: res.update(extra)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ft", "probe"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--corpus", default=os.path.join(HERE, "data", "corpus.jsonl"))
    ap.add_argument("--queries", default=None)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows")
    ap.add_argument("--save", action="store_true", help="save final fine-tuned weights to models/final_<name>.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    out_path = os.path.join(HERE, "results", f"{a.name}.json"); os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = load_corpus(a.corpus)
    if a.limit: rows = rows[:a.limit]
    labels = sorted({r["label"] for r in rows}); lid = {l: i for i, l in enumerate(labels)}
    tr, dv, te = split_rows(rows, a.seed)
    log(f"rows {len(rows)} labels {len(labels)} split {len(tr)}/{len(dv)}/{len(te)}")
    Xtr, Xdv, Xte = texts_of(tr), texts_of(dv), texts_of(te)
    ytr, ydv, yte = (np.array([lid[r["label"]] for r in s]) for s in (tr, dv, te))
    queries = None
    if a.queries and os.path.exists(a.queries):
        queries = [json.loads(l) for l in open(a.queries) if l.strip()]
        queries_all = [q for q in queries if q["label"] in lid]
        queries = [q for q in queries_all if not q.get("ambiguous")]  # main transfer metric: unambiguous questions only
        log(f"queries {len(queries)} clear / {len(queries_all)} incl. ambiguous")

    tok = AutoTokenizer.from_pretrained(a.model)
    enc = AutoModel.from_pretrained(a.model)
    n_params = sum(p.numel() for p in enc.parameters())
    t0 = time.time()

    if a.arm == "probe":
        from sklearn.linear_model import LogisticRegression
        Etr, Edv, Ete = (embed(enc, tok, X, a.max_len, 32) for X in (Xtr, Xdv, Xte))
        log(f"embedded in {time.time()-t0:.0f}s")
        best = None
        for C in (0.3, 1.0, 3.0, 10.0, 30.0):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Etr, ytr)
            f = macro_f1(clf.predict(Edv), ydv, len(labels))
            if best is None or f > best[0]: best = (f, C, clf)
        f, C, clf = best; log(f"probe best C={C} dev macroF1={f:.3f}")
        ldv, lte = clf.decision_function(Edv), clf.decision_function(Ete)
        if len(labels) == 2: ldv = np.stack([-ldv, ldv], 1); lte = np.stack([-lte, lte], 1)
        # latency: encoder forward batch 1
        ts = []
        for x in Xte[:20]:
            t = time.perf_counter(); embed(enc, tok, [x], a.max_len, 1); ts.append(time.perf_counter() - t)
        res = evaluate(a.name, ldv, ydv, lte, yte, len(labels), labels,
                       {"arm": "probe", "model": a.model, "C": C, "params": n_params, "ms_per_example_bs1_torch": 1000 * float(np.median(ts)), "train_seconds": time.time() - t0})
        if queries:
            Eq = embed(enc, tok, [q["text"] for q in queries], a.max_len, 32); yq = np.array([lid[q["label"]] for q in queries])
            lq = clf.decision_function(Eq)
            if len(labels) == 2: lq = np.stack([-lq, lq], 1)
            res["queries"] = evaluate(a.name, ldv, ydv, lq, yq, len(labels), labels)
    else:
        model = Clf(enc, len(labels))
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(Xtr) / a.bs); total = steps_per_epoch * a.epochs; warm = max(1, int(0.06 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)
        start_ep, best_f1, best_state, step = 0, -1.0, None, 0
        ck_dir = os.path.join(HERE, "models"); os.makedirs(ck_dir, exist_ok=True)
        for ep in range(a.epochs - 1, -1, -1):
            ck = os.path.join(ck_dir, f"ckpt_{a.name}_ep{ep}.pt")
            if os.path.exists(ck):
                st = torch.load(ck, map_location="cpu"); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
                start_ep, best_f1, best_state, step = ep + 1, st["best_f1"], st["best_state"], st["step"]
                log(f"resumed from {ck} (next epoch {start_ep})"); break
        idx = list(range(len(Xtr)))
        for ep in range(start_ep, a.epochs):
            model.train(); rng = random.Random(a.seed + ep); rng.shuffle(idx); tl = 0.0; te0 = time.time()
            for bi in range(0, len(idx), a.bs):
                b = idx[bi:bi + a.bs]
                encd = tok([Xtr[i] for i in b], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
                logits = model(encd["input_ids"], encd["attention_mask"])
                loss = F.cross_entropy(logits, torch.tensor(ytr[b]))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1; tl += loss.item()
                if (bi // a.bs) % 20 == 0: log(f"ep{ep} step {bi//a.bs}/{steps_per_epoch} loss {loss.item():.3f}")
            ldv = predict_logits(model, tok, Xdv, a.max_len, 32); f = macro_f1(ldv.argmax(1), ydv, len(labels))
            log(f"ep{ep} done train_loss {tl/steps_per_epoch:.3f} dev macroF1 {f:.3f} acc {(ldv.argmax(1)==ydv).mean():.3f} ({time.time()-te0:.0f}s)")
            if f > best_f1: best_f1, best_state = f, {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best_f1": best_f1, "best_state": best_state, "step": step}, os.path.join(ck_dir, f"ckpt_{a.name}_ep{ep}.pt"))
        if best_state: model.load_state_dict(best_state)
        train_s = time.time() - t0
        ldv = predict_logits(model, tok, Xdv, a.max_len, 32); lte = predict_logits(model, tok, Xte, a.max_len, 32)
        ts = []
        for x in Xte[:20]:
            t = time.perf_counter(); predict_logits(model, tok, [x], a.max_len, 1); ts.append(time.perf_counter() - t)
        res = evaluate(a.name, ldv, ydv, lte, yte, len(labels), labels,
                       {"arm": "ft", "model": a.model, "epochs": a.epochs, "lr": a.lr, "max_len": a.max_len, "params": n_params, "best_dev_macro_f1": best_f1,
                        "ms_per_example_bs1_torch": 1000 * float(np.median(ts)), "train_seconds": train_s})
        if queries:
            lq = predict_logits(model, tok, [q["text"] for q in queries], a.max_len, 32); yq = np.array([lid[q["label"]] for q in queries])
            res["queries"] = evaluate(a.name, ldv, ydv, lq, yq, len(labels), labels)
            with open(os.path.join(HERE, "results", f"{a.name}_query_preds.jsonl"), "w") as f_:
                for q, l in zip(queries, softmax(lq, res["temperature"])):
                    f_.write(json.dumps({"text": q["text"], "label": q["label"], "pred": labels[int(l.argmax())], "conf": float(l.max())}) + "\n")
        if a.save: torch.save({"state": model.state_dict(), "labels": labels, "model": a.model, "max_len": a.max_len}, os.path.join(HERE, "models", f"final_{a.name}.pt"))
        for ep in range(a.epochs):
            ck = os.path.join(HERE, "models", f"ckpt_{a.name}_ep{ep}.pt")
            if os.path.exists(ck): os.remove(ck)
    res["labels"] = labels; res["split"] = {"train": len(tr), "dev": len(dv), "test": len(te)}
    json.dump(res, open(out_path, "w"), indent=1)
    log(f"acc {res['accuracy']:.3f} macroF1 {res['macro_f1']:.3f} ece {res['ece_raw']:.3f}->{res['ece_temp']:.3f} cov@5% {res['coverage_5pct']['coverage']:.2f}" + (f" | queries acc {res['queries']['accuracy']:.3f} f1 {res['queries']['macro_f1']:.3f}" if queries else ""))
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
