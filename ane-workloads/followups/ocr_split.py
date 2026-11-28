#!/usr/bin/env python3
"""Separate converter loss from recognizer loss in an ocr_ane.py run.

  python3 -I followups/ocr_split.py results/<run>/ocr work/ocr/ds/gold followups/ocr_split.json

Every gold 'present' (plain text) and 'table_cell' test is looked up two ways: in Vision's
raw output for the page (what the recognizer produced) and in the adapted markdown the grader
saw (what the conversion script passed on). Found in raw but not in markdown means the
conversion script lost it; found in neither means the recognizer never produced it.

Matching is a normalised substring (case, punctuation, markdown, HTML and whitespace removed),
so it is more lenient than the grader's max_diffs=0 test: both columns are upper bounds on the
benchmark score, not the score. Latex and number tests have no plain text to look for and are
skipped. Table cells are checked together with their named neighbours (up, down, left, right,
top_heading, left_heading); the grader also checks the cell's headings, which this does not.
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

RUN, GOLD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NEIGHBOURS = ("up", "down", "left", "right", "top_heading", "left_heading")


def norm(s):
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[*_`#|>~\\\[\]]", " ", s)
    s = re.sub(r"[^\w]+", " ", s.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def raw_text(arm, row):
    if arm == "text":
        return " ".join(l["text"] for l in row["lines"])
    out = []
    for d in row["documents"]:
        out.append(d.get("title") or "")
        out += [p.get("text", "") for p in d.get("paragraphs", [])]
        for t in d.get("tables", []):
            out += [c.get("text", "") for c in t.get("cells", [])]
        for l in d.get("lists", []):
            out += [i.get("text", "") for i in l.get("items", [])]
    return " ".join(out)


def text_of(v):
    return v.get("text") if isinstance(v, dict) else v


result = {"run": str(RUN), "arms": {}}
for arm in ("text", "docs"):
    cnt = collections.Counter()
    skipped = 0
    tables = collections.Counter()
    with gzip.open(RUN / arm / "raw.jsonl.gz", "rt") as f:
        rows = [json.loads(l) for l in f]
    for row in rows:
        pid = row["page_id"]
        rawn = norm(raw_text(arm, row))
        mdn = norm(json.loads((RUN / arm / "pages" / f"{pid}.json").read_text()).get("markdown", ""))
        tests = json.loads((GOLD / f"{pid}.json").read_text())["tests"]
        has_cells = False
        for t in tests:
            a = t["args"]
            if t["test_type"] == "present":
                texts = [text_of(a["content"])]
                kind = "present"
            elif t["test_type"] == "table_cell":
                texts = [text_of(a["cell"])] + [text_of(a[k]) for k in NEIGHBOURS if k in a]
                kind = "table_cell"
                has_cells = True
            else:
                continue
            if not all(isinstance(x, str) and norm(x) for x in texts):
                skipped += 1
                continue
            wants = [norm(x) for x in texts]
            in_raw, in_md = all(w in rawn for w in wants), all(w in mdn for w in wants)
            cnt[(kind, "raw")] += in_raw
            cnt[(kind, "markdown")] += in_md
            cnt[(kind, "converter_lost")] += in_raw and not in_md
            cnt[(kind, "n")] += 1
        if arm == "docs" and has_cells:
            tables["pages_with_table_tests"] += 1
            tables["pages_with_vision_table"] += any(d.get("tables") for d in row["documents"])
    r = {"skipped_no_plain_text": skipped, "table_pages": dict(tables)}
    for kind in ("present", "table_cell"):
        n = cnt[(kind, "n")]
        r[kind] = {"n": n, "found_in_raw": cnt[(kind, "raw")], "found_in_markdown": cnt[(kind, "markdown")],
                   "converter_lost": cnt[(kind, "converter_lost")]}
    result["arms"][arm] = r
OUT.write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
