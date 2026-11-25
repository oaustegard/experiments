"""Stage 0/1/2 of the MSD vocabulary extension of linkup-sparseup-embed-v2, plus the encodes that score them.

    python3 train.py --stages 0,1,2 --out OUT [--steps1 N] [--steps2 N] [--smoke]

Stage 0: the 696 concept rows added (mean of each catalog name's pieces), no training.
Stage 1: only the new rows (input embedding = tied output row) and their decoder bias train; the rest is frozen,
         so nothing the model knew can be lost.
Stage 2: the whole model trains, new rows at a higher rate, with the original model (frozen, original
         vocabulary, no rewrite) as a teacher: KL between in-batch score distributions keeps its ranking behaviour.
Training pairs come from cache/train_pubmed.txt (PubMed abstracts naming each concept; no evaluation abstract):
the query is a surface form of a concept the abstract is about (catalog name, HGNC symbol, or, for concepts
outside the test split, the MeSH heading), or the abstract's title; in-batch negatives, other abstracts
carrying the same concept masked out. Loss: cross-entropy over dot-product scores plus FLOPS regularisation.
After each stage: the fixture's abstracts and query forms, and SciFact (general retrieval), are encoded and
saved as sparse matrices in OUT for score.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

from msd_vocab import ConceptTokenizer, extend_model

HEAD = ("logit_shift", "logit_sigma", "position_top_k", "vocab_fold", "query_prefix", "document_prefix",
        "add_prefix_space", "query_max_length", "doc_max_length", "auto_map", "architectures")


def load_model(hf, device):
    raw = json.loads((Path(hf) / "config.json").read_text())
    conf = ModernBertConfig(**{k: v for k, v in raw.items() if k not in HEAD})
    conf._attn_implementation = "sdpa"
    m = ModernBertForMaskedLM(conf)
    sd = load_file(str(Path(hf) / "model.safetensors"))
    fold = sd.pop("vocab_fold_index").long()
    m.load_state_dict(sd, strict=False)
    m.tie_weights()
    return m.to(device), fold.to(device), raw


class Splade(torch.nn.Module):
    def __init__(self, model, fold, shift=18.0, sigma=1.0, k=12):
        super().__init__()
        self.m, self.fold, self.shift, self.sigma, self.k = model, fold, shift, sigma, k

    def forward(self, ids, attn, pool):
        logits = self.m(input_ids=ids, attention_mask=attn).logits.float()
        w = torch.log1p(F.relu(logits - self.shift) / self.sigma)
        cut = w.topk(self.k, dim=-1).values[..., -1:]
        w = w * (w >= cut).detach() * pool.unsqueeze(-1)
        v = w.max(dim=1).values
        return torch.zeros_like(v).scatter_reduce(1, self.fold[: v.shape[1]].expand_as(v), v, reduce="amax", include_self=False)


def batchify(items, device, pad=50283):
    L = max(len(i) for i, _ in items)
    ids = torch.full((len(items), L), pad, dtype=torch.long)
    attn = torch.zeros((len(items), L), dtype=torch.long)
    pool = torch.zeros((len(items), L), dtype=torch.float)
    for r, (i, p) in enumerate(items):
        ids[r, : len(i)] = torch.tensor(i)
        attn[r, : len(i)] = 1
        pool[r, : len(p)] = torch.tensor(p, dtype=torch.float)
    return ids.to(device), attn.to(device), pool.to(device)


@torch.no_grad()
def encode_all(sp_model, items, device, budget=4096):
    """Batches of at most `budget` padded tokens: the head materialises [tokens, 51,066] logits."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    out = [None] * len(items)
    sp_model.eval()
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(items[i][0]) > budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    for idx in groups:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            v = sp_model(*batchify([items[i] for i in idx], device))
        v = v.float().cpu().numpy()
        for r, i in enumerate(idx):
            out[i] = sp.csr_matrix(v[r: r + 1])
    return sp.vstack(out).tocsr()


def read_corpus(path):
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\n").split(" ||| ")
        if len(p) == 4:
            docs.append({"pmid": p[0], "title": p[1], "mesh": [m for m in p[2].split(";") if m], "abstract": p[3]})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="hf")
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="out")
    ap.add_argument("--stages", default="0,1,2")
    ap.add_argument("--steps1", type=int, default=1200)
    ap.add_argument("--steps2", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--doc-len", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    D = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tdev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else dev)
    random.seed(0)
    torch.manual_seed(0)
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

    vocab = json.loads((D / "vocab.json").read_text())
    fx = json.loads((D / "fixture.json").read_text())
    concepts_tsv = [line.split("\t") for line in (D / "concepts.tsv").read_text().splitlines()[1:]]
    test_idx = {int(r[0]) for r in concepts_tsv if r[1] == "test"}
    mesh_map = json.loads((D / "mesh_map.json").read_text())
    tok = AutoTokenizer.from_pretrained(a.hf)
    ct = ConceptTokenizer(tok, vocab)

    model, fold0, raw = load_model(a.hf, dev)
    V0 = extend_model(model, ct)
    fold = torch.cat([fold0, torch.arange(V0, V0 + ct.n_new, device=dev)])
    student = Splade(model, fold, raw["logit_shift"], raw["logit_sigma"], raw["position_top_k"])

    # ---- evaluation inputs: fixture docs + query forms; SciFact
    fx_docs = [d["title"] + " " + d["abstract"] for d in fx["docs"]]
    fx_queries = sorted({q for c in fx["concepts"] for q in c["forms"].values()})
    scifact = json.loads((D / "scifact.json").read_text()) if (D / "scifact.json").exists() else None
    if a.smoke:
        fx_docs, fx_queries = fx_docs[:8], fx_queries[:8]
        scifact = None

    def evaluate(tag, rewrite=True):
        t0 = time.time()
        enc = lambda texts, kind, ml: [ct.encode(t, kind, ml, rewrite=rewrite) for t in texts]
        sp.save_npz(out / f"{tag}_fx_docs.npz", encode_all(student, enc(fx_docs, "document", 512), dev))
        sp.save_npz(out / f"{tag}_fx_queries.npz", encode_all(student, enc(fx_queries, "query", 512), dev))
        if scifact:
            sp.save_npz(out / f"{tag}_sf_docs.npz", encode_all(student, enc(scifact["docs"], "document", 512), dev))
            sp.save_npz(out / f"{tag}_sf_queries.npz", encode_all(student, enc(scifact["queries"], "query", 512), dev))
        log("encoded", tag, f"{time.time() - t0:.0f}s")

    (out / "fx_queries.json").write_text(json.dumps(fx_queries, ensure_ascii=False))
    stages = [int(s) for s in a.stages.split(",")]
    if 0 in stages:
        evaluate("s0_norewrite", rewrite=False)   # original vocabulary path (new rows never used): the baseline on GPU
        evaluate("s0")

    if not ({1, 2} & set(stages)):
        return
    # ---- training pairs
    docs = read_corpus(D / "train_pubmed.txt")
    per = {}
    for line in (D / "train_pmids.tsv").read_text().splitlines():
        cid, ids = line.split("\t")
        for p in ids.split():
            per.setdefault(p, cid)
    mesh_of = {ct.cids[int(k)]: v["mesh"] for k, v in mesh_map.items() if int(k) not in test_idx}
    surfaces = {c: list(dict.fromkeys(vocab["concepts"][c]["catalog_names"] + [vocab["concepts"][c]["symbol"]] * bool(vocab["concepts"][c]["symbol"])))
                for c in ct.cids}
    examples = []
    for d in docs:
        text = d["title"] + " " + d["abstract"]
        tags = {c for _, _, c in ct.tagger.spans(text)}
        target = per.get(d["pmid"])
        if target not in tags:
            target = next(iter(tags), None)
        d_ids = ct.encode(text, "document", a.doc_len)
        examples.append({"doc": d_ids, "tags": tags, "target": target, "title": d["title"], "raw": text})
    log("training docs", len(examples), "with a concept", sum(e["target"] is not None for e in examples))

    def make_query(e):
        """-> (query ids, concept or None, raw query string). A title query a quarter of the time or when no concept."""
        c = e["target"]
        if c is None or random.random() < 0.25:
            return ct.encode(e["title"], "query", 64), None, e["title"]
        forms = surfaces[c] + ([mesh_of[c]] * 2 if c in mesh_of else [])
        s = random.choice(forms)
        return ct.encode(s, "query", 64), c, s

    def batches(n_steps):
        for _ in range(n_steps):
            b = random.sample(examples, a.bs)
            yield b, [make_query(e) for e in b]

    def loss_fn(q, d, b, qs, teacher_scores=None):
        scores = q @ d.T
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, (_, c, _) in enumerate(qs):
            if c is None:
                continue
            for j, e in enumerate(b):
                if j != i and c in e["tags"]:
                    mask[i, j] = True
        scores = scores.masked_fill(mask, -1e4)
        loss = F.cross_entropy(scores, torch.arange(len(b), device=scores.device))
        flops = 3e-4 * (q.mean(0) ** 2).sum() + 1e-4 * (d.mean(0) ** 2).sum()
        kl = torch.zeros((), device=scores.device)
        if teacher_scores is not None:
            kl = F.kl_div(F.log_softmax(scores, -1), F.softmax(teacher_scores.masked_fill(mask, -1e4), -1), reduction="batchmean")
        return loss, flops, kl

    model.gradient_checkpointing_enable()
    emb = model.get_input_embeddings().weight
    teacher = None
    if 2 in stages:
        tmodel, tfold, _ = load_model(a.hf, tdev)
        for p in tmodel.parameters():
            p.requires_grad_(False)
        teacher = Splade(tmodel.eval(), tfold, raw["logit_shift"], raw["logit_sigma"], raw["position_top_k"])

    for stage in (1, 2):
        if stage not in stages:
            continue
        steps = 3 if a.smoke else (a.steps1 if stage == 1 else a.steps2)
        if stage == 1:   # only the new rows learn: gradients on the old rows and old biases are zeroed
            for p in model.parameters():
                p.requires_grad_(False)
            emb.requires_grad_(True)
            model.decoder.bias.requires_grad_(True)
            keep = torch.zeros(emb.shape[0], 1, device=dev)
            keep[V0:] = 1
            hooks = [emb.register_hook(lambda g, k=keep: g * k),
                     model.decoder.bias.register_hook(lambda g, k=keep: g * k[:, 0])]
            opt = torch.optim.AdamW([emb, model.decoder.bias], lr=2e-3, weight_decay=0.0)
            lr_at = lambda s: 1.0
        else:
            for h in hooks if 1 in stages else []:
                h.remove()
            for p in model.parameters():
                p.requires_grad_(True)
            others = [p for p in model.parameters() if p is not emb and p is not model.decoder.bias]
            opt = torch.optim.AdamW([{"params": others, "lr": 2e-5}, {"params": [emb, model.decoder.bias], "lr": 2e-4}],
                                    weight_decay=0.01)
            lr_at = lambda s, n=steps: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / n)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
        model.train()
        t0 = time.time()
        for step, (b, qs) in enumerate(batches(steps)):
            ts = None
            if stage == 2:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=tdev.type == "cuda"):
                    tq = teacher(*batchify([ct.encode(s, "query", 64, rewrite=False) for _, _, s in qs], tdev))
                    td = teacher(*batchify([ct.encode(e["raw"], "document", a.doc_len, rewrite=False) for e in b], tdev))
                    ts = (tq.float() @ td.float().T).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                q = student(*batchify([x for x, _, _ in qs], dev))
                d = student(*batchify([e["doc"] for e in b], dev))
            loss, flops, kl = loss_fn(q.float(), d.float(), b, qs, ts)
            total = loss + flops + 0.5 * kl
            if not math.isfinite(total.item()):
                raise RuntimeError(f"non-finite loss at stage {stage} step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 50 == 0 or step == steps - 1:
                log(f"stage{stage} step {step}/{steps} loss {loss.item():.3f} flops {flops.item():.4f} kl {kl.item():.3f} "
                    f"q_nnz {(q > 0).sum(1).float().mean():.0f} d_nnz {(d > 0).sum(1).float().mean():.0f} {time.time() - t0:.0f}s")
        model.eval()
        evaluate(f"s{stage}")
        if stage == 1:
            torch.save({"rows": emb[V0:].detach().cpu(), "bias": model.decoder.bias[V0:].detach().cpu()}, out / "s1_new_rows.pt")
        else:
            torch.save({k: v.detach().cpu().half() for k, v in model.state_dict().items()}, out / "s2_model_fp16.pt")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
