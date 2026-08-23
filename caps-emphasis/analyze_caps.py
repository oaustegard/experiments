"""Measure all-caps usage in a sampled SYNTH corpus.

Doc text = query + synthetic_reasoning + synthetic_answer (the generated
training content).  query_seed_text (verbatim Wikipedia) and constraints
(generation metadata) are measured separately so they can't contaminate the
headline numbers.
"""
import gzip, json, random, re, sys
from collections import Counter

SAMPLE = sys.argv[1]
OUT = sys.argv[2]

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
ROMAN = re.compile(r"^[IVXLCDM]+$")

CURATED_ACRONYMS = set("""
US USA UK EU UN USSR NATO UNESCO UNICEF WHO NASA FBI CIA NSA CDC FDA EPA IRS
DNA RNA ATP PH HIV AIDS MRI CT PCR EKG ECG BMI CPR IQ
CPU GPU RAM ROM HTML XML JSON CSS SQL API URL URI HTTP HTTPS FTP TCP IP DNS
PDF PNG JPEG GIF USB LED LCD OLED AI ML NLP LLM GPT IDE OS SDK UI UX
TV DVD CD VHS FM AM PM BC AD BCE CE CEO CFO CTO COO HR PR VP MBA PHD BA BS MA MS
GDP USD EUR GBP JPY IPO ATM VAT IRA
WWI WWII WWWI NYC LA DC UAE USSR PRC
OK OKAY ID TBD FAQ ETC EG IE VS AKA ASAP DIY FYI RSVP LOL
NBA NFL MLB NHL FIFA UEFA IOC
ISBN ISSN DOI PDFS SI CGS KG KM CM MM MPH KPH RPM DC AC
SARS COVID MERS EBOLA
""".split())

DIRECTIVES = ["NEVER", "ALWAYS", "MUST", "IMPORTANT", "NOTE", "WARNING",
              "CRITICAL", "REQUIRED", "SHOULD", "AVOID", "DO", "NOT",
              "CAUTION", "DANGER", "ATTENTION", "REMEMBER", "ONLY", "ALL"]
MULTIWORD = ["DO NOT", "MUST NOT", "DO NOT USE", "NEVER USE"]

# ---------------------------------------------------------------- pass 1
docs = 0
docs_with_caps = 0
docs_with_caps_na = 0        # non-acronym, curated-set version
word_tokens = 0
caps_counter = Counter()
lower_counter = Counter()
per_field_words = Counter()
per_field_caps = Counter()
per_exercise = Counter()
per_exercise_caps_docs = Counter()
mw_caps = Counter()
mw_lower = Counter()
seed_words = seed_caps = 0
constraints_words = constraints_caps = 0
caps_len_hist = Counter()

FIELDS = ("query", "synthetic_reasoning", "synthetic_answer")


def doc_text(r):
    return "\n\n".join((r.get(f) or "") for f in FIELDS)


with gzip.open(SAMPLE, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        docs += 1
        ex = r.get("exercise") or "?"
        per_exercise[ex] += 1
        has_caps = False
        has_caps_na = False
        for f in FIELDS:
            t = r.get(f) or ""
            for w in WORD.findall(t):
                per_field_words[f] += 1
                word_tokens += 1
                if len(w) >= 2 and w.isupper():
                    caps_counter[w] += 1
                    per_field_caps[f] += 1
                    caps_len_hist[min(len(w), 12)] += 1
                    has_caps = True
                    if not ROMAN.match(w) and w not in CURATED_ACRONYMS:
                        has_caps_na = True
                elif w.islower():
                    lower_counter[w] += 1
        docs_with_caps += has_caps
        docs_with_caps_na += has_caps_na
        if has_caps:
            per_exercise_caps_docs[ex] += 1
        for txt, tag in ((r.get("query_seed_text") or "", "seed"),
                         (r.get("constraints") or "", "constraints")):
            ws = WORD.findall(txt)
            c = sum(1 for w in ws if len(w) >= 2 and w.isupper())
            if tag == "seed":
                seed_words += len(ws); seed_caps += c
            else:
                constraints_words += len(ws); constraints_caps += c
        full = doc_text(r)
        low = full.lower()
        for p in MULTIWORD:
            mw_caps[p] += full.count(p)
            mw_lower[p] += low.count(p.lower())

# ---------------------------------------------------------------- derived
caps_total = sum(caps_counter.values())

def is_acronymish(w):
    """Curated set, roman numeral, or a caps token whose lowercase form is
    essentially absent from the corpus (so it is not a word being shouted)."""
    if w in CURATED_ACRONYMS or ROMAN.match(w):
        return True
    lf = lower_counter.get(w.lower(), 0)
    return lf < 10 or lf < caps_counter[w] * 0.05

caps_nonacr_curated = sum(c for w, c in caps_counter.items()
                          if not (w in CURATED_ACRONYMS or ROMAN.match(w)))
caps_nonacr_data = sum(c for w, c in caps_counter.items() if not is_acronymish(w))

directive_counts = {}
for d in DIRECTIVES:
    directive_counts[d] = {"caps": caps_counter.get(d, 0),
                           "lower": lower_counter.get(d.lower(), 0)}
for p in MULTIWORD:
    directive_counts[p] = {"caps": mw_caps[p],
                           "lower": max(mw_lower[p] - mw_caps[p], 0)}

# ---------------------------------------------------------------- pass 2: bucket 200 occurrences
random.seed(7)
RESERVOIR = 200
res = []
seen = 0
CODEISH = re.compile(r"[_<>{}\[\]|`=/\\]|^\s{4,}")
SPECIAL_TOK = re.compile(r"^[A-Z][A-Z0-9_]*$")

with gzip.open(SAMPLE, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        for f in FIELDS:
            t = r.get(f) or ""
            for m in WORD.finditer(t):
                w = m.group()
                if len(w) < 2 or not w.isupper():
                    continue
                seen += 1
                if len(res) < RESERVOIR:
                    res.append((r.get("synth_id"), f, t, m.start(), m.end(), w))
                else:
                    j = random.randrange(seen)
                    if j < RESERVOIR:
                        res[j] = (r.get("synth_id"), f, t, m.start(), m.end(), w)

buckets = Counter()
examples = {}


def classify(text, s, e, w):
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    if le < 0:
        le = len(text)
    line = text[ls:le]
    before = text[ls:s]
    after = text[e:le]
    line_stripped = line.strip()
    # inside inline code / fenced block?
    fence_before = text.count("```", 0, s)
    in_code = fence_before % 2 == 1
    tick_before = line.count("`", 0, s - ls)
    if in_code or tick_before % 2 == 1 or line.count("|") >= 2:
        return "structured-marker", line
    if "_" in text[max(0, s - 1):min(len(text), e + 1)] or CODEISH.search(before[-3:] + after[:3]):
        # e.g. MAX_LEN, <TOKEN>, [NOTE]
        if re.search(r"[_<>\[\]{}]", before[-2:] + after[:2]):
            return "structured-marker", line
    if is_acronymish(w):
        return "acronym/initialism", line
    line_letters = [c for c in line_stripped if c.isalpha()]
    all_caps_line = bool(line_letters) and all(c.isupper() for c in line_letters)
    line_initial = before.strip(" \t#*->•-–—") == ""
    if all_caps_line and len(line_letters) > 2:
        return "heading/label", line
    if line_initial and (after.lstrip().startswith(":") or line_stripped.endswith(":")
                         or before.strip().startswith(("#", "*", "-", "•"))):
        return "heading/label", line
    if after.lstrip().startswith(":"):
        return "heading/label", line
    n_words = len(WORD.findall(line))
    has_lower = any(c.islower() for c in line)
    if has_lower and n_words >= 5 and not line_initial:
        return "emphasis-in-sentence", line
    if has_lower and n_words >= 5 and line_initial:
        return "emphasis-in-sentence", line
    return "other", line


for sid, f, t, s, e, w in res:
    b, line = classify(t, s, e, w)
    buckets[b] += 1
    snippet = t[max(0, s - 90):e + 90].replace("\n", " ⏎ ")
    examples.setdefault(b, [])
    if len(examples[b]) < 6:
        examples[b].append({"token": w, "field": f, "snippet": snippet[:200]})

result = {
    "dataset_id": "PleIAs/SYNTH",
    "config": "default",
    "split": "train",
    "source": "https://datasets-server.huggingface.co/rows (partial index: "
              "1,089,584 of 79,648,272 rows exposed); random offsets, seed 20260823",
    "text_fields_measured": list(FIELDS),
    "n_documents": docs,
    "n_word_tokens": word_tokens,
    "n_caps_tokens_total": caps_total,
    "doc_level": {
        "frac_docs_with_any_caps_token": docs_with_caps / docs,
        "frac_docs_with_nonacronym_caps_token": docs_with_caps_na / docs,
        "n_docs_with_any_caps": docs_with_caps,
        "n_docs_with_nonacronym_caps": docs_with_caps_na,
    },
    "token_level": {
        "caps_per_1000_words_all": 1000 * caps_total / word_tokens,
        "caps_per_1000_words_excl_curated_acronyms_and_roman":
            1000 * caps_nonacr_curated / word_tokens,
        "caps_per_1000_words_excl_data_driven_acronyms":
            1000 * caps_nonacr_data / word_tokens,
        "n_caps_excl_curated": caps_nonacr_curated,
        "n_caps_excl_data_driven": caps_nonacr_data,
        "caps_token_length_hist": dict(sorted(caps_len_hist.items())),
    },
    "per_field": {f: {"words": per_field_words[f], "caps": per_field_caps[f],
                      "caps_per_1000": 1000 * per_field_caps[f] / max(per_field_words[f], 1)}
                  for f in FIELDS},
    "other_fields": {
        "query_seed_text": {"words": seed_words, "caps": seed_caps,
                            "caps_per_1000": 1000 * seed_caps / max(seed_words, 1)},
        "constraints": {"words": constraints_words, "caps": constraints_caps,
                        "caps_per_1000": 1000 * constraints_caps / max(constraints_words, 1)},
    },
    "per_exercise": {k: {"docs": v, "docs_with_caps": per_exercise_caps_docs[k],
                         "frac": per_exercise_caps_docs[k] / v}
                     for k, v in per_exercise.most_common()},
    "top60_caps_tokens": [{"token": w, "count": c,
                           "lower_form_count": lower_counter.get(w.lower(), 0),
                           "acronymish": bool(is_acronymish(w))}
                          for w, c in caps_counter.most_common(60)],
    "directive_words": directive_counts,
    "buckets": {"n_sampled_occurrences": len(res), "counts": dict(buckets),
                "examples": examples},
}

with open(OUT, "w") as fh:
    json.dump(result, fh, indent=2, ensure_ascii=False)
print(json.dumps({k: result[k] for k in
                  ("n_documents", "n_word_tokens", "n_caps_tokens_total",
                   "doc_level", "token_level")}, indent=2)[:2000])
print("buckets", dict(buckets))
