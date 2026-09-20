"""Tokenizer fragmentation audit: how many pieces does each tokenizer spend on the site's own terms?

Domain terms = word types in data/corpus.jsonl that (a) contain a digit, a hyphen, or ≥2 capitals, or are ≥9
letters, (b) occur ≥3 times, (c) are absent from a general-English frequency list (wordfreq top 50k) so the
list is the site's vocabulary rather than ordinary long words. Control = the 300 most frequent ordinary
words in the same corpus that ARE in the general list.
Reports pieces/term per tokenizer, share of terms split into ≥3 and ≥5 pieces, and the 40 most frequent
domain terms with their piece counts. Writes results/vocab_audit.json. Completion line: 'done -> <path>'."""
import json, os, re, collections, sys
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKS = {
    "ModernBERT/Ettin BPE (50k)": os.path.join(HERE, "models", "ettin-encoder-32m"),
    "gte-small WordPiece (30k, uncased)": os.path.join(HERE, "models", "gte-small"),
    "BiomedBERT PubMed WordPiece (30k, uncased)": os.path.join(HERE, "models", "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
    "BioClinical-ModernBERT (ModernBERT BPE)": os.path.join(HERE, "models", "BioClinical-ModernBERT-base"),
}
try:
    from wordfreq import top_n_list
    general = set(top_n_list("en", 50000))
except Exception:
    general = None
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "corpus.jsonl"))]
cnt = collections.Counter()
for r in rows:
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-/]*[A-Za-z0-9]", (r.get("title") or "") + " " + (r.get("text") or "")):
        cnt[w] += 1

def is_domain(w):
    if general is not None and w.lower() in general: return False
    caps = sum(c.isupper() for c in w)
    return bool(re.search(r"\d", w)) or "-" in w or caps >= 2 or len(w) >= 9

domain = [(w, c) for w, c in cnt.most_common() if c >= 3 and is_domain(w)]
control = [(w, c) for w, c in cnt.most_common() if general is not None and w.lower() in general and w.isalpha() and len(w) >= 4][:300]
print(f"corpus word types {len(cnt)}, domain terms {len(domain)}, control words {len(control)}, general list {'ok' if general else 'MISSING'}")

def pieces(tok, w):
    return len(tok.tokenize(" " + w if "BPE" in name or "ModernBERT" in name else w))

res = {"n_domain_terms": len(domain), "n_control": len(control), "tokenizers": {}, "top_terms": []}
piece_table = {}
for name, path in TOKS.items():
    tok = AutoTokenizer.from_pretrained(path)
    pd = [pieces(tok, w) for w, _ in domain]; pc = [pieces(tok, w) for w, _ in control]
    wd = sum(c for _, c in domain); freq_weighted = sum(p * c for p, (_, c) in zip(pd, domain)) / wd
    res["tokenizers"][name] = {
        "vocab_size": len(tok), "domain_pieces_per_term": sum(pd) / len(pd), "domain_pieces_per_term_freq_weighted": freq_weighted,
        "domain_share_ge3": sum(p >= 3 for p in pd) / len(pd), "domain_share_ge5": sum(p >= 5 for p in pd) / len(pd),
        "control_pieces_per_word": sum(pc) / len(pc), "control_share_ge3": sum(p >= 3 for p in pc) / len(pc),
    }
    piece_table[name] = dict(zip([w for w, _ in domain], pd))
    print(f"{name:44s} vocab {len(tok):6d} | domain {sum(pd)/len(pd):.2f} pieces/term (freq-weighted {freq_weighted:.2f}), ≥3: {100*sum(p>=3 for p in pd)/len(pd):.0f}%, ≥5: {100*sum(p>=5 for p in pd)/len(pd):.0f}% | control {sum(pc)/len(pc):.2f}, ≥3: {100*sum(p>=3 for p in pc)/len(pc):.0f}%")
for w, c in domain[:40]:
    res["top_terms"].append({"term": w, "count": c, **{n: piece_table[n][w] for n in TOKS}})
short = list(TOKS)
print("\nmost frequent domain terms (count | pieces under each tokenizer, order as above):")
for t in res["top_terms"]:
    print(f"  {t['term']:34s} {t['count']:5d} | " + "  ".join(str(t[n]) for n in short))
out = os.path.join(HERE, "results", "vocab_audit.json"); json.dump(res, open(out, "w"), indent=1)
print(f"done -> {out}")
