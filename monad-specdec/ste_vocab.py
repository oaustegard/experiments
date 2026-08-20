"""How many BPE tokens does a ~900-word controlled vocabulary span?

ASD-STE100's approved general dictionary is roughly a thousand words. A draft
head is indexed by BPE tokens, not words, so the question is how many token
types those words decompose into, and how much of real text that token set
covers once technical names — which STE deliberately leaves unbounded — are
included.

The official dictionary is copyrighted and distributed only on request, so this
uses the highest-frequency word types of a general corpus as a stand-in. That
substitution is defensible on STE's own terms: the STEMG says the approved words
were chosen for "simplicity, flexibility and frequency of use".
"""
import collections, glob, json, re
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")

corpus = []
for p in sorted(glob.glob("/workspace/experiments/*/RESULTS.md"))[:40]:
    corpus.append(open(p, errors="ignore").read())
blob = "\n".join(corpus)

words = re.findall(r"[A-Za-z]+", blob.lower())
wfreq = collections.Counter(words)

rows = []
for K in [500, 900, 1500, 3000]:
    top = [w for w, _ in wfreq.most_common(K)]
    # A word is written both sentence-initially and mid-sentence, so a draft
    # head must carry both casings and the leading-space variant.
    pieces = set()
    for w in top:
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            pieces.update(tok(form, add_special_tokens=False).input_ids)
    word_cov = sum(wfreq[w] for w in top) / len(words)
    rows.append({"approved_words": K,
                 "bpe_token_types": len(pieces),
                 "tokens_per_word": round(len(pieces) / K, 2),
                 "word_occurrence_coverage": round(word_cov, 4)})
    print(rows[-1], flush=True)

# What the alphabetic-word view misses entirely.
ids = tok(blob, add_special_tokens=False).input_ids
alpha = sum(1 for i in ids if re.fullmatch(r"\s*[A-Za-z]+", tok.decode([i])))
out = {"rows": rows,
       "corpus_word_tokens": len(words),
       "corpus_bpe_tokens": len(ids),
       "bpe_tokens_that_are_plain_words": round(alpha / len(ids), 4),
       "note": ("Everything outside that fraction — punctuation, digits, code, "
                "identifiers, subword fragments of technical names — is vocabulary "
                "a word list cannot supply.")}
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
json.dump(out, open("ste_vocab.json", "w"), indent=2)
