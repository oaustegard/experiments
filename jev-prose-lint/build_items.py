#!/usr/bin/env python3
"""Build the item set: declaude_lint hits on muninn.austegard.com blog posts, with context.

Each item is one lint hit (density hits excluded) with the paragraph it sits in, the
register.md entry for its category, and a revision signal from git history:
  removed - the hit appeared in an earlier committed version and is gone from the final one
  kept    - the hit is in the final published version

    python3 build_items.py            # writes data/hits_all.jsonl and data/items.jsonl
"""
import collections
import hashlib
import html
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke  # noqa: E402

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
SITE = spoke("muninn.austegard.com")
LINT = Path("/mnt/skills/user/declauding/scripts/declaude_lint.py")
REGISTER = Path("/mnt/skills/user/declauding/references/register.md")
SKIP = {"index.html", "_template.html"}
BLOCKS = ["p", "li", "h1", "h2", "h3", "h4", "blockquote", "figcaption", "td", "dd", "summary"]
SEED = 20261107
TARGET = 300
CAP_PER_CATEGORY = 45

# lint category -> register.md entry numbers
ENTRY = {
    "staging": [1], "negation-first": [2], "significance": [3], "agency": [4], "deferred-noun": [5],
    "locator": [6], "header": [7], "cadence": [9, 15], "rhetorical-q": [10], "aphorism": [12],
    "self-grading": [13], "em-dash": [16], "rtfm": [18], "dev-cliche": [19], "slop": [20],
    "time-inflation": [22], "editorializing": [23], "copula": [24], "participle": [25], "triad": [26],
    "false-range": [28], "diff-anchored": [34], "subjectless": [35], "announce": [38], "spec-ese": [40],
    "flat-certainty": [43], "juridical": [44], "verification": [45], "privative": [46],
    "exhaustive": [47], "candour": [48], "reversal": [49], "totalizing": [51],
}


def git(*args):
    return subprocess.run(["git", "-C", str(SITE), *args], capture_output=True, text=True, check=True).stdout


def register_entries():
    text = REGISTER.read_text()
    out = {}
    for m in re.finditer(r"^## (\d+)\. .*?(?=^## |\Z)", text, re.S | re.M):
        out[int(m.group(1))] = m.group(0).strip()
    return out


def lint(text):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(text)
    r = subprocess.run(["python3", str(LINT), "--json", "--no-html", fh.name], capture_output=True, text=True)
    Path(fh.name).unlink()
    return [h for h in json.loads(r.stdout)["hits"] if h.get("line", 0) > 0]


def plain(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def key(h):
    return (h["category"], plain(h["match"]))


def paragraph(soup, line, match):
    """Text of the block element nearest `line` that contains the match."""
    want = plain(match)
    best = None
    for tag in soup.find_all(BLOCKS):
        if tag.sourceline is None or tag.find(BLOCKS):
            continue
        txt = " ".join(tag.get_text(" ").split())
        if want and want in txt:
            d = abs(tag.sourceline - line)
            if best is None or d < best[0]:
                best = (d, txt, tag.name)
    return (best[1], best[2]) if best else (None, None)


def main():
    entries = register_entries()
    posts = sorted(p for p in (SITE / "blog").glob("*.html") if p.name not in SKIP)
    rows = []
    for post in posts:
        rel = f"blog/{post.name}"
        final = post.read_text(errors="replace")
        soup = BeautifulSoup(final, "html.parser")
        final_hits = lint(final)
        final_keys = {key(h) for h in final_hits}
        shas = git("log", "--format=%H", "--", rel).split()
        removed = {}
        for sha in shas[1:]:  # older versions, newest first; shas[0] is the final version
            try:
                old = git("show", f"{sha}:{rel}")
            except subprocess.CalledProcessError:
                continue
            for h in lint(old):
                k = key(h)
                if k not in final_keys and k not in removed:
                    oldsoup = BeautifulSoup(old, "html.parser")
                    para, tag = paragraph(oldsoup, h["line"], h["match"])
                    removed[k] = {**h, "paragraph": para, "tag": tag, "sha": sha[:10]}
        for h in final_hits:
            para, tag = paragraph(soup, h["line"], h["match"])
            rows.append({**h, "post": post.name, "revision": "kept", "paragraph": para, "tag": tag})
        for h in removed.values():
            rows.append({**h, "post": post.name, "revision": "removed"})
    for r in rows:
        r["match_text"] = plain(r["match"])
        r["entry_ids"] = ENTRY.get(r["category"], [])
        r["entry"] = "\n\n".join(entries[i] for i in r["entry_ids"] if i in entries)
        r["id"] = hashlib.sha1(f"{r['post']}|{r['category']}|{r['match_text']}|{r['revision']}".encode()).hexdigest()[:10]
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / "hits_all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    usable = [r for r in rows if r["paragraph"] and r["entry"]]
    print(f"hits {len(rows)} usable {len(usable)} "
          f"revision {collections.Counter(r['revision'] for r in rows)}")
    # Sample: every removed hit (capped per category), then kept hits stratified by category.
    rng = random.Random(SEED)
    rng.shuffle(usable)
    picked, per_cat = [], collections.Counter()
    seen = set()
    for r in sorted(usable, key=lambda r: r["revision"] != "removed"):
        k = (r["post"], r["category"], r["match_text"])
        if k in seen or per_cat[r["category"]] >= CAP_PER_CATEGORY:
            continue
        if r["revision"] == "removed" and sum(x["revision"] == "removed" for x in picked) >= TARGET // 2:
            continue
        seen.add(k)
        picked.append(r)
        per_cat[r["category"]] += 1
        if len(picked) >= TARGET:
            break
    items = [{k: r[k] for k in ("id", "post", "category", "note", "match_text", "paragraph", "tag",
                                "entry_ids", "entry", "revision")} for r in picked]
    (data / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    print(f"items {len(items)} revision {collections.Counter(r['revision'] for r in items)}")
    print(f"categories {per_cat.most_common()}")


if __name__ == "__main__":
    main()
