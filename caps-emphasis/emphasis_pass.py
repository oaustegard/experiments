"""Second pass: isolate genuinely emphatic caps from all-caps blocks/acronyms."""
import gzip, json, re, sys
from collections import Counter

SAMPLE, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
ROMAN = re.compile(r"^[IVXLCDM]+$")
base = json.load(open(BASE))
lower_counter = {t["token"].lower(): t["lower_form_count"] for t in base["top60_caps_tokens"]}
CUR = set(base["top60_caps_tokens"][0].keys())  # unused; reuse rule below


CURATED = set(open("curated_acronyms.txt").read().split())


def acronymish(w, lf, cf):
    """Acronym unless its lowercase form is a common corpus word that is
    overwhelmingly written lowercase (i.e. this caps form is the word shouted)."""
    if w in CURATED or ROMAN.match(w):
        return True
    return not (lf >= 200 and cf < 0.05 * lf)


# need global lower/caps counts -> recompute cheaply
capsC, lowC = Counter(), Counter()
FIELDS = ("query", "synthetic_reasoning", "synthetic_answer")
rows = []
with gzip.open(SAMPLE, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        rows.append(r)
        for f in FIELDS:
            for w in WORD.findall(r.get(f) or ""):
                if len(w) >= 2 and w.isupper():
                    capsC[w] += 1
                elif w.islower():
                    lowC[w] += 1

ACR = {w: bool(acronymish(w, lowC.get(w.lower(), 0), c)) for w, c in capsC.items()}

emph = Counter()
emph_examples = {}
n_emph = 0
words_total = 0
shout_docs = 0
shout_caps = 0
shout_words = 0
DIR = {"NEVER", "ALWAYS", "MUST", "IMPORTANT", "NOTE", "WARNING", "CRITICAL",
       "REQUIRED", "SHOULD", "AVOID", "DO", "NOT", "ONLY", "ALL", "EVERY", "ANY"}
dir_ctx = {d: [] for d in DIR}

for r in rows:
    doc = "\n\n".join((r.get(f) or "") for f in FIELDS)
    letters = [c for c in doc if c.isalpha()]
    upfrac = sum(c.isupper() for c in letters) / max(len(letters), 1)
    ws = WORD.findall(doc)
    words_total += len(ws)
    is_shout = upfrac > 0.5
    if is_shout:
        shout_docs += 1
        shout_words += len(ws)
        shout_caps += sum(1 for w in ws if len(w) >= 2 and w.isupper())
        continue
    for line in doc.split("\n"):
        ll = [c for c in line if c.isalpha()]
        if not ll:
            continue
        if all(c.isupper() for c in ll):     # whole-line caps -> heading, not emphasis
            continue
        lw = WORD.findall(line)
        if len(lw) < 5:
            continue
        for m in WORD.finditer(line):
            w = m.group()
            if len(w) < 2 or not w.isupper() or ACR.get(w, True):
                continue
            n_emph += 1
            emph[w] += 1
            if len(emph_examples.setdefault(w, [])) < 3:
                emph_examples[w].append(line.strip()[:200])
            if w in DIR and len(dir_ctx[w]) < 5:
                dir_ctx[w].append(line.strip()[:200])

out = {
    "definition": "emphatic caps = all-caps token (len>=2) that is not acronym-like, "
                  "sitting on a mixed-case line of >=5 words, in a document that is "
                  "not itself majority-uppercase",
    "n_documents": len(rows),
    "n_word_tokens_all_docs": words_total,
    "all_caps_documents": {"n": shout_docs, "frac": shout_docs / len(rows),
                           "words": shout_words, "caps_tokens": shout_caps,
                           "share_of_all_caps_tokens": shout_caps / base["n_caps_tokens_total"]},
    "n_emphatic_caps": n_emph,
    "emphatic_caps_per_1000_words": 1000 * n_emph / words_total,
    "frac_of_all_caps_tokens_that_are_emphatic": n_emph / base["n_caps_tokens_total"],
    "top_emphatic_tokens": [{"token": w, "count": c, "examples": emph_examples[w][:3]}
                            for w, c in emph.most_common(40)],
    "directive_contexts": {k: v for k, v in dir_ctx.items() if v},
}
json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("n_documents", "all_caps_documents", "n_emphatic_caps",
                                      "emphatic_caps_per_1000_words",
                                      "frac_of_all_caps_tokens_that_are_emphatic")}, indent=2))
print([ (t["token"], t["count"]) for t in out["top_emphatic_tokens"][:30] ])
