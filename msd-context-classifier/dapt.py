"""Domain-adaptive continued pretraining (MLM) of an encoder on the site corpus, optionally with vocabulary expansion.

Usage: python3 dapt.py --model models/ettin-encoder-32m --name dapt_ettin_32m [--expand-vocab N] [--epochs E] [--term-mask P]
Training text = every page in --corpus whose url is NOT in the dev/test split of data/corpus.jsonl (those 270 pages are
held out from adaptation so mlm_ppl.py and train.py remain clean), truncated to --max-len tokens.
--expand-vocab N   add the N most frequent domain terms (count >= 20, >= 3 pieces under the tokenizer) as whole tokens;
                   each new embedding row = mean of the term's old piece embeddings (decoder tied to embeddings).
--term-mask P      in addition to 15% random masking, mask every piece of a domain-term occurrence with probability P
                   (whole-term masking; 0 = plain DAPT).
Checkpoints per epoch to models/<name>/ (save_pretrained + tokenizer); resumes if models/<name>/epoch.json exists.
Log: results/<name>.log. Completion line: 'done -> models/<name>'."""
import argparse, json, os, re, sys, time, math, random, collections
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, split_rows, log

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = r"[A-Za-z][A-Za-z0-9\-/]*[A-Za-z0-9]"
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--corpus", default=None, help="default: data/full/corpus.jsonl if present else data/corpus.jsonl")
ap.add_argument("--expand-vocab", type=int, default=0); ap.add_argument("--term-mask", type=float, default=0.0)
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--seed", type=int, default=20260920); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
out_dir = os.path.join(HERE, "models", a.name); os.makedirs(out_dir, exist_ok=True)

base = load_corpus(os.path.join(HERE, "data", "corpus.jsonl")); _, dv, te = split_rows(base)
held = {r["url"] for r in dv} | {r["url"] for r in te}
corpus = a.corpus or (os.path.join(HERE, "data", "full", "corpus.jsonl") if os.path.exists(os.path.join(HERE, "data", "full", "corpus.jsonl")) else os.path.join(HERE, "data", "corpus.jsonl"))
rows = [r for r in load_corpus(corpus) if r["url"] not in held]
if a.limit: rows = rows[:a.limit]
texts = [((r.get("title") or "") + "\n" + (r.get("text") or "")).strip() for r in rows]
log(f"corpus {corpus}: {len(rows)} pages after holding out {len(held)} dev/test urls")

from wordfreq import top_n_list
general = set(top_n_list("en", 50000)); cnt = collections.Counter()
for t in texts:
    for w in re.findall(WORD, t): cnt[w] += 1
def is_domain(w):
    if w.lower() in general: return False
    return bool(re.search(r"\d", w)) or "-" in w or sum(c.isupper() for c in w) >= 2 or len(w) >= 9
domain = {w for w, c in cnt.items() if c >= 3 and is_domain(w)}

tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForMaskedLM.from_pretrained(a.model)
added = []
if a.expand_vocab:
    cands = [(w, c) for w, c in cnt.most_common() if c >= 20 and is_domain(w) and len(tok.tokenize(" " + w)) >= 3][:a.expand_vocab]
    emb = model.get_input_embeddings().weight.data
    old_ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w, _ in cands]
    n_added = tok.add_tokens([w for w, _ in cands])
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight.data
    for (w, _), ids in zip(cands, old_ids):
        emb[tok.convert_tokens_to_ids(w)] = emb[ids].mean(0)
    added = [w for w, _ in cands]
    log(f"expanded vocab by {n_added} terms (first 12: {added[:12]}); embeddings {tuple(emb.shape)}; tied={model.get_output_embeddings().weight.data_ptr() == emb.data_ptr()}")
json.dump({"added_tokens": added, "args": vars(a), "n_pages": len(rows)}, open(os.path.join(out_dir, "dapt_meta.json"), "w"), indent=1)

mask_id = tok.mask_token_id; special = set(tok.all_special_ids); V = len(tok)
def encode_all(texts):
    encs = tok(texts, truncation=True, max_length=a.max_len, return_offsets_mapping=True)
    out = []
    for i, t in enumerate(texts):
        ids = encs["input_ids"][i]; offs = encs["offset_mapping"][i]
        term_tok = set()
        if a.term_mask > 0:
            for m in re.finditer(WORD, t):
                if m.group() in domain:
                    for j, (s, e) in enumerate(offs):
                        if e > s and s < m.end() and e > m.start(): term_tok.add(j)
        out.append((ids, term_tok))
    return out
data = encode_all(texts); log(f"encoded {len(data)} pages, {sum(len(d[0]) for d in data)} tokens")

def make_batch(items, rng):
    L = max(len(ids) for ids, _ in items); x = torch.full((len(items), L), tok.pad_token_id); y = torch.full((len(items), L), -100); am = torch.zeros((len(items), L), dtype=torch.long)
    for r, (ids, term_tok) in enumerate(items):
        ids_t = torch.tensor(ids); n = len(ids); am[r, :n] = 1
        prob = torch.full((n,), 0.15)
        if term_tok: prob[list(term_tok)] = max(0.15, a.term_mask)
        m = torch.bernoulli(prob).bool()
        for j, t in enumerate(ids):
            if t in special: m[j] = False
        y[r, :n][m] = ids_t[m]
        inp = ids_t.clone(); rnd = torch.rand(n)
        inp[m & (rnd < 0.8)] = mask_id
        rand_tok = torch.randint(0, V, (n,)); sel = m & (rnd >= 0.8) & (rnd < 0.9); inp[sel] = rand_tok[sel]
        x[r, :n] = inp
    return x, am, y

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps_per_epoch = math.ceil(len(data) / a.bs); total = steps_per_epoch * a.epochs; warm = max(1, int(0.06 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
start = 0; ep_file = os.path.join(out_dir, "epoch.json")
if os.path.exists(ep_file):
    st = json.load(open(ep_file)); start = st["next_epoch"]
    model = AutoModelForMaskedLM.from_pretrained(out_dir); tok = AutoTokenizer.from_pretrained(out_dir)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); opt.load_state_dict(torch.load(os.path.join(out_dir, "opt.pt")))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))); sched.load_state_dict(torch.load(os.path.join(out_dir, "sched.pt")))
    log(f"resumed at epoch {start}")
order = list(range(len(data)))
for ep in range(start, a.epochs):
    model.train(); rng = random.Random(a.seed + ep); rng.shuffle(order); order.sort(key=lambda i: len(data[i][0]) // 32)  # length-bucketed
    chunks = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]; rng.shuffle(chunks); tl = 0.0; t0 = time.time()
    for si, ch in enumerate(chunks):
        x, am, y = make_batch([data[i] for i in ch], rng)
        loss = model(input_ids=x, attention_mask=am, labels=y).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); tl += loss.item()
        if si % 25 == 0: log(f"ep{ep} step {si}/{len(chunks)} loss {loss.item():.3f} ({(time.time()-t0)/(si+1):.1f}s/step)")
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir); torch.save(opt.state_dict(), os.path.join(out_dir, "opt.pt")); torch.save(sched.state_dict(), os.path.join(out_dir, "sched.pt"))
    json.dump({"next_epoch": ep + 1, "train_loss": tl / len(chunks), "seconds": time.time() - t0}, open(ep_file, "w"))
    log(f"ep{ep} done train_loss {tl/len(chunks):.3f} ({time.time()-t0:.0f}s) -> {out_dir}")
for f in ("opt.pt", "sched.pt"):
    p = os.path.join(out_dir, f)
    if os.path.exists(p): os.remove(p)
print(f"done -> {out_dir}", flush=True)
