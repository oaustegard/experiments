"""Markdown emphasis markers in SYNTH — the complement to the caps measurement.

Same sample, same fields, same denominators as analyze_caps.py so the /1000
rates are directly comparable.
"""
import gzip, json, random, re, sys
from collections import Counter

SAMPLE, OUT = sys.argv[1], sys.argv[2]
FIELDS = ("query", "synthetic_reasoning", "synthetic_answer")
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

PAT = {
    "bold_asterisk":   re.compile(r"\*\*(?!\s)([^\n*]{1,300}?)(?<!\s)\*\*"),
    "italic_asterisk": re.compile(r"(?<![*\w])\*(?!\s|\*)([^\n*]{1,300}?)(?<!\s)\*(?![*\w])"),
    "bold_underscore": re.compile(r"(?<![_\w])__(?!\s|_)([^\n_]{1,300}?)(?<!\s)__(?![_\w])"),
    "italic_underscore": re.compile(r"(?<![_\w*])_(?!\s|_)([^\n_]{1,300}?)(?<!\s)_(?![_\w])"),
    "code_inline":     re.compile(r"(?<!`)`(?!`)([^\n`]{1,300}?)`(?!`)"),
    "code_fence":      re.compile(r"```[^\n]*\n.*?```", re.S),
}
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+\S", re.M)
BOLD_LINE_INITIAL = re.compile(r"^\s{0,6}(?:[-*+•]\s+|\d+\.\s+)?\*\*(?!\s)([^\n*]{1,120}?)\*\*", re.M)
BOLD_LABEL_COLON = re.compile(r"^\s{0,6}(?:[-*+•]\s+|\d+\.\s+)?\*\*(?!\s)([^\n*]{1,120}?)\*\*\s*:", re.M)

DIRECTIVES = ["never", "always", "must", "not", "important", "critical", "note",
              "do not", "required", "avoid", "warning", "caution", "remember",
              "only", "all", "should", "key", "no"]
# caps counts measured in analyze_caps.py, for the head-to-head table
CAPS_COUNTS = {"never": 5, "always": 6, "must": 2, "not": 256, "important": 0,
               "critical": 2, "note": 2, "do not": 3, "required": 0, "avoid": 0,
               "warning": 1, "caution": 0, "remember": 0, "only": 6, "all": 112,
               "should": 1, "key": None, "no": None}

docs = 0
word_tokens = 0
per_field_words = Counter()
counts = Counter()
per_field_counts = Counter()          # (marker, field)
docs_with = Counter()
headings = 0
heading_h1 = heading_h2 = 0
docs_with_heading = 0
bold_line_initial = 0
bold_label_colon = 0
docs_with_bold_label = 0
bold_content = Counter()
dir_exact = Counter()      # **never**
dir_contains = Counter()   # **... never ...**
lower_counts = Counter()   # plain lowercase occurrences, for share denominators
bold_spans = []            # reservoir
seen_spans = 0
random.seed(11)
RES = 200

with gzip.open(SAMPLE, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        docs += 1
        doc_has = set()
        doc_head = False
        doc_lbl = False
        for f in FIELDS:
            t = r.get(f) or ""
            ws = WORD.findall(t)
            per_field_words[f] += len(ws)
            word_tokens += len(ws)
            for w in ws:
                if w.islower():
                    lower_counts[w] += 1
            for name, p in PAT.items():
                n = 0
                for m in p.finditer(t):
                    n += 1
                    if name == "bold_asterisk":
                        inner = m.group(1)
                        bold_content[inner.strip().lower()[:60]] += 1
                        low = inner.lower().strip(" :*_")
                        for d in DIRECTIVES:
                            if low == d:
                                dir_exact[d] += 1
                            if re.search(r"(?<![a-z])" + re.escape(d) + r"(?![a-z])", inner.lower()):
                                dir_contains[d] += 1
                        seen_spans += 1
                        rec = (f, t, m.start(), m.end(), inner)
                        if len(bold_spans) < RES:
                            bold_spans.append(rec)
                        else:
                            j = random.randrange(seen_spans)
                            if j < RES:
                                bold_spans[j] = rec
                if n:
                    counts[name] += n
                    per_field_counts[(name, f)] += n
                    doc_has.add(name)
            hs = HEADING.findall(t)
            headings += len(hs)
            heading_h1 += sum(1 for h in hs if len(h) == 1)
            heading_h2 += sum(1 for h in hs if len(h) == 2)
            if hs:
                doc_head = True
            bi = len(BOLD_LINE_INITIAL.findall(t))
            bc = len(BOLD_LABEL_COLON.findall(t))
            bold_line_initial += bi
            bold_label_colon += bc
            if bc:
                doc_lbl = True
        for k in doc_has:
            docs_with[k] += 1
        docs_with_heading += doc_head
        docs_with_bold_label += doc_lbl

# ------------------------------------------------------------------ buckets
BULLET_MARK = re.compile(r"[●○◐☑✓✗⚠※∴→←≠≈±∈∀∃⏎]")


def bucket(text, s, e, inner):
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    if le < 0:
        le = len(text)
    line = text[ls:le]
    before, after = text[ls:s], text[e:le]
    if text.count("```", 0, s) % 2 == 1 or line.count("|") >= 2:
        return "structured-marker", line
    if re.search(r"[`_=<>{}\\/]", inner) or re.match(r"^\s*[\d.]+\s*$", inner):
        return "structured-marker", line
    line_initial = re.match(r"^\s{0,6}(?:[-*+•]\s+|\d+\.\s+|#{1,6}\s+)?$", before) is not None
    labelled = after.lstrip().startswith(":") or inner.rstrip().endswith(":")
    if line_initial and (labelled or after.strip() in ("", "●", "○", "◐")):
        return "heading/label", line
    if labelled:
        return "heading/label", line
    if re.match(r"^\s*(is|are|was|were|means|refers to|denotes|=|≡|:=|→|–|—|-)\s",
                after, re.I):
        return "term-definition", line
    nw = len(WORD.findall(line))
    if nw >= 5 and any(c.islower() for c in line):
        return "emphasis-in-sentence", line
    if BULLET_MARK.search(line) or nw < 5:
        return "structured-marker", line
    return "other", line


buckets = Counter()
bex = {}
for f, t, s, e, inner in bold_spans:
    b, line = bucket(t, s, e, inner)
    buckets[b] += 1
    snip = t[max(0, s - 80):e + 80].replace("\n", " ⏎ ")[:200]
    bex.setdefault(b, [])
    if len(bex[b]) < 6:
        bex[b].append({"inner": inner[:80], "field": f, "snippet": snip})

per_1000 = {k: 1000 * v / word_tokens for k, v in counts.items()}
any_md = counts["bold_asterisk"] + counts["italic_asterisk"] + \
    counts["bold_underscore"] + counts["italic_underscore"]

out = {
    "dataset_id": "PleIAs/SYNTH", "config": "default", "split": "train",
    "sample": {"n_documents": docs, "n_word_tokens": word_tokens,
               "note": "same 22,100-doc sample as synth_caps.json; no re-fetch"},
    "marker_counts": dict(counts),
    "marker_per_1000_words": per_1000,
    "marker_doc_fraction": {k: docs_with[k] / docs for k in counts},
    "marker_docs": {k: docs_with[k] for k in counts},
    "all_markdown_emphasis_per_1000": 1000 * any_md / word_tokens,
    "per_field": {f: {"words": per_field_words[f],
                      **{k: per_field_counts[(k, f)] for k in PAT},
                      **{k + "_per_1000": 1000 * per_field_counts[(k, f)] / max(per_field_words[f], 1)
                         for k in PAT}}
                  for f in FIELDS},
    "headings": {"n_atx_headings": headings, "h1": heading_h1, "h2": heading_h2,
                 "per_1000_words": 1000 * headings / word_tokens,
                 "doc_fraction": docs_with_heading / docs},
    "bold_line_initial": {"n": bold_line_initial,
                          "per_1000_words": 1000 * bold_line_initial / word_tokens,
                          "n_bold_label_with_colon": bold_label_colon,
                          "doc_fraction_bold_label": docs_with_bold_label / docs},
    "directive_head_to_head": {
        d: {"bold_exact": dir_exact[d], "bold_containing": dir_contains[d],
            "caps": CAPS_COUNTS[d], "plain_lowercase": lower_counts.get(d, None) if " " not in d else None}
        for d in DIRECTIVES},
    "buckets": {"n_sampled_bold_spans": len(bold_spans), "counts": dict(buckets),
                "examples": bex},
    "top_bold_contents": bold_content.most_common(50),
}
json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
print(json.dumps({"marker_per_1000_words": per_1000,
                  "marker_doc_fraction": out["marker_doc_fraction"],
                  "headings": out["headings"],
                  "bold_line_initial": out["bold_line_initial"]}, indent=2))
print("buckets", dict(buckets))
