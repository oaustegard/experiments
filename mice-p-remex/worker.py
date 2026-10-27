"""MICE-P worker. Runs ONLY inside scripts/jail.sh: it loads the model's
trust_remote_code module (modernbert_hf.py from xpmir/Mice-P-*), which nobody
here wrote. No network, no secrets; inputs and outputs are files in the jail dir.

    encode  MODEL DOCS.jsonl OUT_PREFIX          -> OUT_PREFIX.{vecs.npy,offsets.npy,ids.json}
    score   MODEL QUERIES.json CANDS.json VEC_PREFIX OUT.json [DECODED.npy]
    check   MODEL QUERIES.json CANDS.json DOCS.jsonl   (Path A == Path B, cached == uncached)

Documents are tokenized [CLS] d [SEP] to 512 tokens, queries [CLS] q [SEP] to 32,
matching the split the HF CrossEncoder path performs (_split_input_ids_vectorized).
encode stores doc_compress(normalize(center(h))): the 128-d vector MICE-P hands to
FastPLAID. score feeds stored or codec-decoded 128-d vectors back in as doc_hidden.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
Q_LEN, D_LEN = 32, 512


def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa").eval()
    return tok, model


def cache_query_bottom(model):
    """Run the query bottom layers once per query instead of once per pair.

    score() repeats one query across a batch of candidates; every row is identical,
    so row 0 is computed and broadcast. `check` verifies this matches uncached."""
    orig = model._forward_bottom

    def expand(x, b):
        if torch.is_tensor(x):
            return x.expand(b, *x.shape[1:]) if x.dim() > 0 and x.shape[0] == 1 else x
        if isinstance(x, (tuple, list)):
            return type(x)(expand(y, b) for y in x)
        return x

    def fb(seq, mask=None, **kw):
        if (seq.dim() == 2 and seq.shape[0] > 1 and bool((seq == seq[:1]).all())
                and (mask is None or bool((mask == mask[:1]).all())) and not kw.get("cu_seqlens")):
            h, pe = orig(seq[:1], mask=None if mask is None else mask[:1], **kw)
            return expand(h, seq.shape[0]), expand(pe, seq.shape[0])
        return orig(seq, mask=mask, **kw)

    model._forward_bottom = fb
    return orig


@torch.no_grad()
def encode(model_dir, docs_path, out_prefix, bs=16):
    tok, model = load(model_dir)
    docs = [json.loads(l) for l in open(docs_path)]
    order = np.argsort([-len(d["text"]) for d in docs], kind="stable")
    per_doc = [None] * len(docs)
    t0 = time.time()
    for s in range(0, len(docs), bs):
        idx = order[s:s + bs]
        enc = tok([docs[i]["text"] for i in idx], truncation=True, max_length=D_LEN,
                  padding=True, return_tensors="pt")
        h, _ = model.encode_documents(doc_ids=enc["input_ids"], doc_mask=enc["attention_mask"])
        for j, i in enumerate(idx):
            per_doc[i] = h[j][enc["attention_mask"][j].bool()].float().numpy()
        if (s // bs) % 50 == 0:
            print(f"encode {s + len(idx)}/{len(docs)} {time.time() - t0:.0f}s", flush=True)
    lens = np.array([len(x) for x in per_doc])
    np.save(out_prefix + ".vecs.npy", np.concatenate(per_doc).astype(np.float32))
    np.save(out_prefix + ".offsets.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    json.dump([d["id"] for d in docs], open(out_prefix + ".ids.json", "w"))
    print(f"encoded {len(docs)} docs, {lens.sum()} tokens, {time.time() - t0:.0f}s", flush=True)


def pad_docs(vecs, offsets, rows):
    L = max(offsets[r + 1] - offsets[r] for r in rows)
    H = np.zeros((len(rows), L, vecs.shape[1]), dtype=np.float32)
    M = np.zeros((len(rows), L), dtype=np.int64)
    for j, r in enumerate(rows):
        n = offsets[r + 1] - offsets[r]
        H[j, :n] = vecs[offsets[r]:offsets[r + 1]]
        M[j, :n] = 1
    return torch.from_numpy(H), torch.from_numpy(M)


@torch.no_grad()
def score_pairs(tok, model, qtext, vecs, offsets, rows, bs=50):
    q = tok([qtext], truncation=True, max_length=Q_LEN, return_tensors="pt")
    out = []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        H, M = pad_docs(vecs, offsets, r)
        logits = model(query_ids=q["input_ids"].expand(len(r), -1),
                       query_mask=q["attention_mask"].expand(len(r), -1),
                       doc_hidden=H, doc_mask=M).logits
        out.append(logits.reshape(len(r), -1)[:, 0].float().numpy())
    return np.concatenate(out)


def score(model_dir, queries_path, cands_path, vec_prefix, out_path, decoded=None):
    tok, model = load(model_dir)
    cache_query_bottom(model)
    queries = json.load(open(queries_path))
    cands = json.load(open(cands_path))
    vecs = np.load(decoded) if decoded else np.load(vec_prefix + ".vecs.npy")
    offsets = np.load(vec_prefix + ".offsets.npy")
    row = {d: i for i, d in enumerate(json.load(open(vec_prefix + ".ids.json")))}
    res, t0 = {}, time.time()
    for n, (qid, docs) in enumerate(cands.items()):
        s = score_pairs(tok, model, queries[qid], vecs, offsets, [row[d] for d in docs])
        res[qid] = {d: float(x) for d, x in zip(docs, s)}
        if n % 100 == 0:
            print(f"score {n}/{len(cands)} {time.time() - t0:.0f}s", flush=True)
    tmp = out_path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, out_path)
    print(f"scored {len(cands)} queries in {time.time() - t0:.0f}s", flush=True)


class DedupRows(torch.nn.Module):
    """Apply `inner` once when every batch row is the same tensor, then broadcast.

    score_fast batches one document against many queries, so the doc-side
    up-projection, LayerNorm and K/V projection see identical rows."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if x.dim() == 3 and x.shape[0] > 1 and bool((x[1:] == x[:1]).all()):
            y = self.inner(x[:1])
            return y.expand(x.shape[0], *y.shape[1:])
        return self.inner(x)


def install_fast_paths(model, tok, queries_text, bs=64):
    """Precompute every query's bottom-layer states (padded to Q_LEN) and route
    _forward_bottom to them; dedupe the doc-side projections. Returns (ids, mask, set_rows)."""
    enc = tok(queries_text, truncation=True, max_length=Q_LEN, padding="max_length", return_tensors="pt")
    orig = model._forward_bottom
    hs, pe1 = [], None
    with torch.no_grad():
        for s in range(0, len(queries_text), bs):
            h, _ = orig(enc["input_ids"][s:s + bs], mask=enc["attention_mask"][s:s + bs])
            hs.append(h)
        _, pe1 = orig(enc["input_ids"][:1], mask=enc["attention_mask"][:1])
    HQ = torch.cat(hs)
    state = {"rows": None}

    def expand(x, b):
        if torch.is_tensor(x):
            return x.expand(b, *x.shape[1:]) if x.dim() > 0 and x.shape[0] == 1 else x
        if isinstance(x, (tuple, list)):
            return type(x)(expand(y, b) for y in x)
        return x

    def fb(seq, mask=None, **kw):
        rows = state["rows"]
        assert rows is not None and seq.shape[0] == len(rows)
        return HQ[rows], expand(pe1, len(rows))

    model._forward_bottom = fb
    model.doc_decompress = DedupRows(model.doc_decompress)
    for layer in model.top_layers:
        layer.doc_cross_attn_norm = DedupRows(layer.doc_cross_attn_norm)
        layer.cross_attn.Wqkv = DedupRows(layer.cross_attn.Wqkv)
    return enc["input_ids"], enc["attention_mask"], state


@torch.no_grad()
def score_fast(model_dir, queries_path, cands_path, vec_prefix, out_path, decoded=None, bs=64):
    """Same scores as `score`, grouped by document: one doc x all queries that retrieved it."""
    tok, model = load(model_dir)
    queries = json.load(open(queries_path))
    cands = json.load(open(cands_path))
    qids = list(cands)
    vecs = np.load(decoded) if decoded else np.load(vec_prefix + ".vecs.npy")
    offsets = np.load(vec_prefix + ".offsets.npy")
    row = {d: i for i, d in enumerate(json.load(open(vec_prefix + ".ids.json")))}
    t0 = time.time()
    ids, mask, state = install_fast_paths(model, tok, [queries[q] for q in qids])
    by_doc = {}
    for qi, q in enumerate(qids):
        for d in cands[q]:
            by_doc.setdefault(d, []).append(qi)
    res = {q: {} for q in qids}
    for n, (d, qis) in enumerate(by_doc.items()):
        H1, M1 = pad_docs(vecs, offsets, [row[d]])
        for s in range(0, len(qis), bs):
            qs = qis[s:s + bs]
            state["rows"] = qs
            logits = model(query_ids=ids[qs], query_mask=mask[qs],
                           doc_hidden=H1.expand(len(qs), -1, -1), doc_mask=M1.expand(len(qs), -1)).logits
            for qi, x in zip(qs, logits.reshape(len(qs), -1)[:, 0].tolist()):
                res[qids[qi]][d] = x
        if n % 500 == 0:
            print(f"score_fast doc {n}/{len(by_doc)} {time.time() - t0:.0f}s", flush=True)
    tmp = out_path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, out_path)
    print(f"scored {sum(len(v) for v in by_doc.values())} pairs over {len(by_doc)} docs in {time.time() - t0:.0f}s", flush=True)


def check_fast(model_dir, queries_path, cands_path, vec_prefix, ref_scores, n_q=8):
    """score_fast on the first n_q queries must reproduce reference `score` output."""
    cands = json.load(open(cands_path))
    sub = {q: cands[q] for q in list(cands)[:n_q]}
    sub_path = "/tmp/check_fast_cands.json"
    json.dump(sub, open(sub_path, "w"))
    score_fast(model_dir, queries_path, sub_path, vec_prefix, "/tmp/check_fast_out.json")
    got, ref = json.load(open("/tmp/check_fast_out.json")), json.load(open(ref_scores))
    diffs = [abs(got[q][d] - ref[q][d]) for q in sub for d in sub[q]]
    print(f"check_fast: {len(diffs)} pairs, max |diff| {max(diffs):.2e}, mean {np.mean(diffs):.2e}")


@torch.no_grad()
def check(model_dir, queries_path, cands_path, docs_path, n_q=2, n_d=6):
    """Path A (stored 128-d doc_hidden, query cache on) must equal Path B (raw ids, CrossEncoder-style)."""
    tok, model = load(model_dir)
    queries = json.load(open(queries_path))
    cands = json.load(open(cands_path))
    text = {d["id"]: d["text"] for d in map(json.loads, open(docs_path))}
    worst = 0.0
    for qid in list(cands)[:n_q]:
        dids = cands[qid][:n_d]
        q = tok([queries[qid]] * n_d, truncation=True, max_length=Q_LEN, padding=True, return_tensors="pt")
        d = tok([text[x] for x in dids], truncation=True, max_length=D_LEN, padding=True, return_tensors="pt")
        b = model(query_ids=q["input_ids"], query_mask=q["attention_mask"],
                  doc_ids=d["input_ids"], doc_mask=d["attention_mask"]).logits.reshape(n_d, -1)[:, 0]
        h, m = model.encode_documents(doc_ids=d["input_ids"], doc_mask=d["attention_mask"])
        vecs = np.concatenate([h[j][m[j].bool()].numpy() for j in range(n_d)])
        offs = np.concatenate([[0], np.cumsum([int(m[j].sum()) for j in range(n_d)])])
        a_plain = score_pairs(tok, model, queries[qid], vecs, offs, list(range(n_d)))
        orig = cache_query_bottom(model)
        a_cached = score_pairs(tok, model, queries[qid], vecs, offs, list(range(n_d)))
        model._forward_bottom = orig
        # CrossEncoder-style concatenated input, split inside forward
        pair = tok([queries[qid]] * n_d, [text[x] for x in dids], truncation="only_second",
                   max_length=Q_LEN + D_LEN, padding=True, return_tensors="pt")
        ce = model(input_ids=pair["input_ids"], attention_mask=pair["attention_mask"]).logits.reshape(n_d, -1)[:, 0]
        print(qid, "B", np.round(b.numpy(), 4), "\n   A", np.round(a_plain, 4),
              "\n   A cached", np.round(a_cached, 4), "\n   CE pair", np.round(ce.numpy(), 4), flush=True)
        worst = max(worst, float(np.abs(b.numpy() - a_plain).max()), float(np.abs(a_plain - a_cached).max()))
    print("max |diff| A/B/cached:", worst)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"encode": encode, "score": score, "check": check,
     "score_fast": score_fast, "check_fast": check_fast}[cmd](*args)
