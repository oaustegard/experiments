"""Masked-LM diagnostics on held-out domain pages: token-level and whole-term.

Usage: python3 mlm_ppl.py <name> <model_dir> [--pages N] [--rounds K]
Pages = the test split of data/corpus.jsonl (same split as train.py), truncated to 256 tokens.

TOKEN mode (rounds of 15% random masking): CE and top-1 per masked token, split by whether the token
sits inside a domain term. Under a fragmenting tokenizer this mostly measures piece completion
("IL", "-", [MASK] -> "6"), so it flatters the model.

TERM mode (the one that matters): every occurrence of a domain term has ALL its pieces masked at once;
the same for a control set of ordinary words (in the general-English list, >= 4 letters). Up to 8
non-overlapping targets per forward pass. Reports per-term summed CE, per-piece CE, and exact recovery
(every piece argmax-correct) for domain terms vs control words.
Domain terms = vocab_audit.py's definition (regex + not in wordfreq top-50k, count >= 3 in the corpus).
Writes results/mlm_<name>.json. Completion line: 'done -> <path>'."""
import argparse, json, os, re, sys, collections, math, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, split_rows, log

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("model"); ap.add_argument("--pages", type=int, default=138)
ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--seed", type=int, default=20260920)
ap.add_argument("--corpus", default=os.path.join(HERE, "data", "corpus.jsonl"))
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
WORD = r"[A-Za-z][A-Za-z0-9\-/]*[A-Za-z0-9]"

rows = load_corpus(a.corpus); tr, dv, te = split_rows(rows)
pages = [((r.get("title") or "") + "\n" + (r.get("text") or "")).strip() for r in te[:a.pages]]
from wordfreq import top_n_list
general = set(top_n_list("en", 50000)); cnt = collections.Counter()
for r in rows:
    for w in re.findall(WORD, (r.get("title") or "") + " " + (r.get("text") or "")): cnt[w] += 1
def is_domain(w):
    if w.lower() in general: return False
    return bool(re.search(r"\d", w)) or "-" in w or sum(c.isupper() for c in w) >= 2 or len(w) >= 9
domain = {w for w, c in cnt.items() if c >= 3 and is_domain(w)}

tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForMaskedLM.from_pretrained(a.model); model.eval()
mask_id = tok.mask_token_id; rng = np.random.RandomState(a.seed)
tokagg = {"domain": [0.0, 0, 0], "other": [0.0, 0, 0]}
term = {"domain": {"ce_sum": 0.0, "pieces": 0, "n": 0, "exact": 0, "terms": collections.Counter(), "exact_terms": collections.Counter()},
        "control": {"ce_sum": 0.0, "pieces": 0, "n": 0, "exact": 0, "terms": collections.Counter(), "exact_terms": collections.Counter()}}
t0 = time.time()

def spans_to_tokens(offs, s, e):
    return [i for i in range(len(offs)) if offs[i][1] > offs[i][0] and offs[i][0] < e and offs[i][1] > s]

with torch.no_grad():
    for pi, text in enumerate(pages):
        enc = tok(text, truncation=True, max_length=a.max_len, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"][0]; am = enc["attention_mask"]; offs = enc["offset_mapping"][0].tolist(); L = len(ids)
        limit = max(o[1] for o in offs)
        targets = []  # (kind, word, token idx list)
        for m in re.finditer(WORD, text):
            if m.end() > limit: break
            w = m.group(); kind = "domain" if w in domain else ("control" if (w.lower() in general and w.isalpha() and len(w) >= 4) else None)
            if kind is None: continue
            ti = [i for i in spans_to_tokens(offs, m.start(), m.end()) if ids[i].item() not in tok.all_special_ids]
            if ti: targets.append((kind, w, ti))
        # --- token mode
        cand = [i for i in range(L) if offs[i][1] > offs[i][0] and ids[i].item() not in tok.all_special_ids]
        dom_tok = set(i for k, _, ti in targets if k == "domain" for i in ti)
        for _ in range(a.rounds):
            k = max(1, int(0.15 * len(cand))); pick = rng.choice(cand, size=k, replace=False)
            x = ids.clone(); x[pick] = mask_id
            lp = torch.log_softmax(model(input_ids=x.unsqueeze(0), attention_mask=am).logits[0][pick].float(), -1); gold = ids[pick]
            ce = -lp[torch.arange(len(pick)), gold]; top1 = (lp.argmax(-1) == gold)
            for j, i in enumerate(pick):
                key = "domain" if i in dom_tok else "other"; tokagg[key][0] += float(ce[j]); tokagg[key][1] += int(top1[j]); tokagg[key][2] += 1
        # --- term mode: batches of up to 8 non-overlapping targets, all pieces masked
        rng.shuffle(targets)
        for b in range(0, len(targets), 8):
            batch = targets[b:b + 8]; used = set(); keep = []
            for kind, w, ti in batch:
                if any(i in used or i - 1 in used or i + 1 in used for i in ti): continue
                used.update(ti); keep.append((kind, w, ti))
            if not keep: continue
            x = ids.clone()
            for _, _, ti in keep: x[ti] = mask_id
            lp = torch.log_softmax(model(input_ids=x.unsqueeze(0), attention_mask=am).logits[0].float(), -1)
            for kind, w, ti in keep:
                g = ids[ti]; ce = float(-lp[ti, g].sum()); exact = bool((lp[ti].argmax(-1) == g).all())
                T = term[kind]; T["ce_sum"] += ce; T["pieces"] += len(ti); T["n"] += 1; T["exact"] += int(exact); T["terms"][w] += 1; T["exact_terms"][w] += int(exact)
        if pi % 20 == 0: log(f"page {pi}/{len(pages)}: term-CE domain {term['domain']['ce_sum']/max(1,term['domain']['n']):.2f} control {term['control']['ce_sum']/max(1,term['control']['n']):.2f}")

res = {"name": a.name, "model": a.model, "pages": len(pages), "rounds": a.rounds, "max_len": a.max_len, "vocab_size": len(tok), "seconds": time.time() - t0, "token": {}, "term": {}}
for key in tokagg:
    s, c, n = tokagg[key]; res["token"][key] = {"n_masked": n, "ce": s / max(1, n), "top1": c / max(1, n)}
for kind in term:
    T = term[kind]; res["term"][kind] = {"n": T["n"], "n_types": len(T["terms"]), "pieces_per_term": T["pieces"] / max(1, T["n"]), "ce_per_term": T["ce_sum"] / max(1, T["n"]),
                                       "ce_per_piece": T["ce_sum"] / max(1, T["pieces"]), "exact": T["exact"] / max(1, T["n"])}
worst = sorted(((w, T["exact_terms"][w] / c, c) for w, c in term["domain"]["terms"].items() if c >= 3), key=lambda x: (x[1], -x[2]))[:25]
res["term"]["domain_worst_terms"] = [{"term": w, "exact": round(e, 2), "n": c} for w, e, c in worst]
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); out = os.path.join(HERE, "results", f"mlm_{a.name}.json"); json.dump(res, open(out, "w"), indent=1)
d, c = res["term"]["domain"], res["term"]["control"]
log(f"{a.name}: TERM domain CE/term {d['ce_per_term']:.2f} exact {d['exact']:.3f} (n={d['n']}, {d['pieces_per_term']:.1f} pieces) | control CE/term {c['ce_per_term']:.2f} exact {c['exact']:.3f} (n={c['n']}, {c['pieces_per_term']:.1f} pieces) | TOKEN domain CE {res['token']['domain']['ce']:.2f} other {res['token']['other']['ce']:.2f}")
print(f"done -> {out}")
