"""Section embedder: teacher vectors, training and evaluation. Runs on the Colab VM (or any CUDA box).

    python3 run/embed.py teacher                                   # gte-modernbert-base vectors for every text
    python3 run/embed.py train NAME --init jhu-clsp/ettin-encoder-68m --stage distill
    python3 run/embed.py train NAME --init ... --stage joint --from CKPT     # contrastive + teacher pull
    python3 run/embed.py train NAME --init ... --stage contrast               # contrastive alone
    python3 run/embed.py eval NAME [--quant] | eval --baseline gte|capemox|medcpt|e17c

Student: encoder, mean pooling, bias-free linear head to 768-d, unit length. `distill` pulls every
section and query toward the teacher's vector (1 - cosine). `joint` adds an in-batch contrastive loss
over each query's source section and its hard negatives; `contrast` is that loss alone.
Inputs under DATA (default /content/data): sections.jsonl.gz, emb_data.json (built by run/emb_data.py).
Outputs: WORK/ckpt/NAME.pt, results/emb_NAME.json.
"""
import argparse
import gzip
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DETTIN_DATA", "/content/data"))
WORK = Path(os.environ.get("DETTIN_WORK", "/content/work"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_FILE = os.environ.get("DETTIN_EMB", "emb_data.json")   # a second eval set swaps this file
TEACHER = "Alibaba-NLP/gte-modernbert-base"
DOC_LEN, Q_LEN, SCALE = 1024, 64, 30.0


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda")


def load_sections():
    with gzip.open(DATA / "sections.jsonl.gz", "rt") as f:
        return {r["sid"]: r["text"] for r in map(json.loads, f)}


class Emb(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.enc = AutoModel.from_pretrained(init, attn_implementation="sdpa")
        self.head = nn.Linear(self.enc.config.hidden_size, 768, bias=False)

    def forward(self, ids, att):
        with amp():
            h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).to(h.dtype)
        pooled = ((h * m).sum(1) / m.sum(1)).float()
        return F.normalize(self.head(pooled), dim=-1)


def tok_batch(tok, texts, max_len):
    b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return b["input_ids"].to(DEV), b["attention_mask"].to(DEV)


@torch.no_grad()
def encode(model, tok, texts, max_len, bs=64):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for i in range(0, len(order), bs):
        idx = order[i: i + bs]
        ids, att = tok_batch(tok, [texts[j] for j in idx], max_len)
        out[idx] = model(ids, att).float().cpu().numpy()
    return out


TOK_OF = {"capemox/ettin-encoder-68m-contrastive-2.5m": "jhu-clsp/ettin-encoder-68m"}   # saved under transformers 5


@torch.no_grad()
def st_encode(name, texts, max_len, bs=64, pool="cls", prefix=""):
    """A baseline or the teacher through plain transformers: CLS or mean pooling, unit length."""
    tok = AutoTokenizer.from_pretrained(TOK_OF.get(name, name))
    model = AutoModel.from_pretrained(name).to(DEV).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = None
    for i in range(0, len(order), bs):
        idx = order[i: i + bs]
        b = tok([prefix + texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        with amp():
            h = model(**b).last_hidden_state
        if pool == "cls":
            v = h[:, 0]
        else:
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)
        v = F.normalize(v.float(), dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
        out[idx] = v
    del model
    torch.cuda.empty_cache()
    return out


def teacher():
    sec = load_sections()
    d = json.loads((DATA / EMB_FILE).read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    sids = sorted(sec)
    t0 = time.time()
    v = st_encode(TEACHER, [sec[s] for s in sids], DOC_LEN, bs=48)
    np.save(WORK / "teacher_docs.npy", v.astype(np.float16))
    (WORK / "teacher_docs.json").write_text(json.dumps(sids))
    qs = [p["query"] for p in d["train_pairs"]]
    np.save(WORK / "teacher_queries.npy", st_encode(TEACHER, qs, Q_LEN, bs=256).astype(np.float16))
    print(f"teacher: {len(sids)} sections, {len(qs)} queries, {time.time() - t0:.0f}s", flush=True)


def sched(opt, total):
    warm = max(1, int(0.06 * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))


def train(a):
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    sec = load_sections()
    d = json.loads((DATA / EMB_FILE).read_text())
    tok = AutoTokenizer.from_pretrained(a.init)
    model = Emb(a.init).to(DEV)
    if a.gc:   # the 68M joint stage at batch 24 does not fit a 40 GB A100 without it
        model.enc.gradient_checkpointing_enable()
    if a.from_ckpt:
        model.load_state_dict(torch.load(WORK / "ckpt" / f"{a.from_ckpt}.pt", map_location=DEV))
    tsid = json.loads((WORK / "teacher_docs.json").read_text())
    trow = {s: i for i, s in enumerate(tsid)}
    tdoc = np.load(WORK / "teacher_docs.npy", mmap_mode="r")
    tq = np.load(WORK / "teacher_queries.npy", mmap_mode="r")

    def tvec(rows, src):
        return torch.tensor(np.stack([src[i] for i in rows]).astype(np.float32), device=DEV)

    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=0.01)
    log, t0, step = [], time.time(), 0
    if a.stage == "distill":
        items = [("d", s) for s in d["distill_sids"]] + [("q", i) for i in range(len(d["train_pairs"]))]
        total = math.ceil(len(items) / a.bs) * a.epochs
        sc = sched(opt, total)
        for _ in range(a.epochs):
            rng.shuffle(items)
            # batches of similar length keep padding down; the batch order is shuffled
            chunks = [sorted(items[i: i + a.bs * 50], key=lambda x: len(sec[x[1]]) if x[0] == "d" else 0)
                      for i in range(0, len(items), a.bs * 50)]
            batches = [c[i: i + a.bs] for c in chunks for i in range(0, len(c), a.bs)]
            rng.shuffle(batches)
            model.train()
            for b in batches:
                docs = [x[1] for x in b if x[0] == "d"]
                qi = [x[1] for x in b if x[0] == "q"]
                loss = 0.0
                if docs:
                    ids, att = tok_batch(tok, [sec[s] for s in docs], DOC_LEN)
                    loss = loss + (1 - (model(ids, att) * tvec([trow[s] for s in docs], tdoc)).sum(-1)).sum()
                if qi:
                    ids, att = tok_batch(tok, [d["train_pairs"][i]["query"] for i in qi], Q_LEN)
                    loss = loss + (1 - (model(ids, att) * tvec(qi, tq)).sum(-1)).sum()
                loss = loss / len(b)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sc.step()
                step += 1
                log.append(loss.item())
                if step % 100 == 0:
                    print(f"[{a.name}] step {step}/{total} loss {np.mean(log[-100:]):.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    else:
        lam = 1.0 if a.stage == "joint" else 0.0
        pairs = list(range(len(d["train_pairs"])))
        total = math.ceil(len(pairs) / a.bs) * a.epochs
        sc = sched(opt, total)
        for _ in range(a.epochs):
            rng.shuffle(pairs)
            model.train()
            for i in range(0, len(pairs), a.bs):
                b = [d["train_pairs"][j] for j in pairs[i: i + a.bs]]
                if len(b) < 2:
                    continue
                docs, seen = [], {}
                for p in b:
                    for s in [p["pos"]] + p["negs"]:
                        if s not in seen:
                            seen[s] = len(docs)
                            docs.append(s)
                target = torch.tensor([seen[p["pos"]] for p in b], device=DEV)
                qids, qatt = tok_batch(tok, [p["query"] for p in b], Q_LEN)
                dids, datt = tok_batch(tok, [sec[s] for s in docs], DOC_LEN)
                qv, dv = model(qids, qatt), model(dids, datt)
                # a query's own negatives stay negatives; another query's positive or negatives in the batch also count
                loss_c = F.cross_entropy(qv @ dv.T * SCALE, target)
                loss = loss_c
                if lam:
                    pull = (1 - (dv * tvec([trow[s] for s in docs], tdoc)).sum(-1)).mean() + \
                           (1 - (qv * tvec(pairs[i: i + a.bs], tq)).sum(-1)).mean()
                    loss = loss + lam * pull
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sc.step()
                step += 1
                log.append(loss_c.item())
                if step % 50 == 0:
                    print(f"[{a.name}] step {step}/{total} nce {np.mean(log[-50:]):.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    (WORK / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WORK / "ckpt" / f"{a.name}.pt")
    (WORK / "ckpt" / f"{a.name}.json").write_text(json.dumps({
        "init": a.init, "stage": a.stage, "from": a.from_ckpt, "steps": step, "lr": a.lr, "bs": a.bs,
        "epochs": a.epochs, "loss_last100": float(np.mean(log[-100:])), "sec": time.time() - t0}))
    print(f"[{a.name}] TRAIN_DONE steps {step} {(time.time() - t0) / 60:.1f} min", flush=True)


@torch.no_grad()
def fake_quant(model):
    """Round the encoder to what pack.py's t4-w8L63 stores: int8 per output column in +-63 for every
    matmul weight, the token table to 4 bits in groups of 32 with an fp16 scale. The 768-d head stays fp32."""
    for name, mod in model.enc.named_modules():
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in ("Wqkv", "Wo", "Wi"):
            W = mod.weight
            s = W.abs().amax(1, keepdim=True) / 63 + 1e-12
            mod.weight.copy_(torch.clamp(torch.round(W / s), -63, 63) * s)
    T = model.enc.embeddings.tok_embeddings.weight
    V, H = T.shape
    g = T.view(V, H // 32, 32)
    s = g.abs().amax(2, keepdim=True) / 7 + 1e-12
    T.copy_((torch.clamp(torch.round(g / s), -8, 7) * s.half().float()).view(V, H))


def ndcg10(scores, rel):
    """scores: [n_q, n_d] float; rel: list of {doc index: gain}."""
    top = np.argsort(-scores, axis=1)[:, :10]
    out = []
    for i, r in enumerate(rel):
        dcg = sum(r.get(int(dn), 0) / math.log2(k + 2) for k, dn in enumerate(top[i]))
        ideal = sum(g / math.log2(k + 2) for k, g in enumerate(sorted(r.values(), reverse=True)[:10]))
        out.append(dcg / ideal if ideal else 0.0)
    return out


def beir(name):
    from datasets import load_dataset
    corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus")
    queries = load_dataset(f"mteb/{name}", "queries", split="queries")
    qrels = load_dataset(f"mteb/{name}", "default", split="test")
    dids = list(corpus["_id"])
    dpos = {x: i for i, x in enumerate(dids)}
    docs = [((t or "") + " " + (x or "")).strip() for t, x in zip(corpus["title"], corpus["text"])]
    rel = {}
    for q, dd, s in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
        if s > 0 and dd in dpos:
            rel.setdefault(q, {})[dpos[dd]] = s
    qtext = dict(zip(queries["_id"], queries["text"]))
    qids = [q for q in rel if q in qtext]
    return docs, [qtext[q] for q in qids], [rel[q] for q in qids]


def evaluate(a):
    sec = load_sections()
    d = json.loads((DATA / EMB_FILE).read_text())
    name = (a.name or f"base-{a.baseline}") + a.suffix
    if a.baseline:
        spec = {"gte": (TEACHER, TEACHER, "cls", 8192), "capemox": ("capemox/ettin-encoder-68m-contrastive-2.5m",) * 2 + ("mean", 256),
                "medcpt": ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", "cls", 512),
                "bge-small": ("BAAI/bge-small-en-v1.5",) * 2 + ("cls", 512)}[a.baseline]
        qpre = "Represent this sentence for searching relevant passages: " if a.baseline == "bge-small" else ""

        def enc_q(texts):
            return st_encode(spec[0], texts, 64, bs=256, pool=spec[2], prefix=qpre)

        def enc_d(texts, max_len):
            return st_encode(spec[1], texts, min(max_len, spec[3]), bs=48, pool=spec[2])
    else:
        name = a.name + ("-q" if a.quant else "") + a.suffix
        meta = json.loads((WORK / "ckpt" / f"{a.name}.json").read_text())
        tok = AutoTokenizer.from_pretrained(meta["init"])
        model = Emb(meta["init"]).to(DEV)
        model.load_state_dict(torch.load(WORK / "ckpt" / f"{a.name}.pt", map_location=DEV))
        if a.quant:
            fake_quant(model)

        def enc_q(texts):
            return encode(model, tok, texts, Q_LEN, bs=256)

        def enc_d(texts, max_len):
            return encode(model, tok, texts, max_len, bs=64)
    res, t0 = {"name": name}, time.time()
    pool = d["eval_pool"]
    ppos = {s: i for i, s in enumerate(pool)}
    art = [s.split(":")[0] for s in pool]
    D = enc_d([sec[s] for s in pool], DOC_LEN)
    res["sections_per_s"] = round(len(pool) / (time.time() - t0), 1)
    Q = enc_q([q["query"] for q in d["eval_queries"]])
    S = Q @ D.T
    nd = ndcg10(S, [{ppos[q["pos"]]: 1} for q in d["eval_queries"]])
    rank = [(S[i] > S[i, ppos[q["pos"]]]).sum() + 1 for i, q in enumerate(d["eval_queries"])]
    within = []
    for i, q in enumerate(d["eval_queries"]):
        same = [j for j, x in enumerate(art) if x == q["pos"].split(":")[0]]
        within.append(float(S[i, ppos[q["pos"]]] >= S[i, same].max()))
    res["sections"] = {"ndcg@10": float(np.mean(nd)), "recall@1": float(np.mean([r <= 1 for r in rank])),
                       "recall@10": float(np.mean([r <= 10 for r in rank])), "within_article_top1": float(np.mean(within)),
                       "n_queries": len(nd), "pool": len(pool)}
    res["sections_perquery_ndcg"] = [round(x, 4) for x in nd]
    by_style = {}
    for x, q in zip(nd, d["eval_queries"]):
        by_style.setdefault(q.get("style", "?"), []).append(x)
    res["sections_by_style"] = {k: float(np.mean(v)) for k, v in by_style.items()}
    print(f"[{name}] sections {json.dumps(res['sections'])}", flush=True)
    for b in ([] if a.no_beir else ["scifact", "nfcorpus", "trec-covid"]):
        try:
            docs, qs, rel = beir(b)
            Sb = enc_q(qs) @ enc_d(docs, 512).T
            nb = ndcg10(Sb, rel)
            res[b] = {"ndcg@10": float(np.mean(nb)), "n_queries": len(nb), "n_docs": len(docs)}
            res[f"{b}_perquery_ndcg"] = [round(x, 4) for x in nb]
            print(f"[{name}] {b} {res[b]}", flush=True)
        except Exception as e:  # noqa: BLE001  a dataset that fails to load must not lose the section result
            res[b] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"[{name}] {b} failed: {res[b]['error']}", flush=True)
    res["sec"] = time.time() - t0
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"emb_{name}.json").write_text(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("teacher")
    t = sub.add_parser("train")
    t.add_argument("name")
    t.add_argument("--init", required=True)
    t.add_argument("--stage", choices=["distill", "joint", "contrast"], required=True)
    t.add_argument("--from", dest="from_ckpt")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--bs", type=int, default=48)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--gc", action="store_true", help="gradient checkpointing in the encoder")
    e = sub.add_parser("eval")
    e.add_argument("name", nargs="?")
    e.add_argument("--baseline")
    e.add_argument("--quant", action="store_true")
    e.add_argument("--no-beir", action="store_true")
    e.add_argument("--suffix", default="", help="appended to the results file name")
    a = ap.parse_args()
    if a.cmd == "teacher":
        teacher()
    elif a.cmd == "train":
        train(a)
    else:
        evaluate(a)

if __name__ == "__main__":
    main()
