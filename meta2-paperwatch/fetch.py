#!/usr/bin/env python3
"""Fetch the full text of every ledger paper with read == "full" into cache/text/<id>.txt.

HTML first (https://arxiv.org/html/<id>v1), PDF fallback via pdftotext. The reference
list is cut and the text capped at MAX_CHARS so no call crosses Haiku's 100K-token
long-prompt rate card. Paper text is data: nothing here executes it.
"""
import json, re, subprocess, sys, time
from pathlib import Path

import bs4

HERE = Path(__file__).resolve().parent
SEEN = HERE.parent / "paper-watch" / "seen.json"
OUT = HERE / "cache" / "text"
MAX_CHARS = 160_000
MIN_HTML_CHARS = 30_000


def html_to_text(html):
    s = bs4.BeautifulSoup(html, "lxml")
    for m in s.find_all("math"):
        m.replace_with(" " + (m.get("alttext") or "") + " ")
    for t in s(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    for a in s.find_all("a", href=True):  # keep external link targets: code/weights checks need them
        h = a["href"]
        if h.startswith("http") and "arxiv.org/html" not in h and h not in a.get_text():
            a.append(f" <{h}>")
    for t in s.find_all(class_=re.compile(r"ltx_bibliography")):
        t.decompose()
    root = s.find("article") or s.body or s
    t = root.get_text("\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def cut_refs(t):
    m = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", t)
    if m and m.start() > len(t) * 0.4:
        tail = t[m.end():]
        app = re.search(r"\n\s*(Appendix|APPENDIX|A\s+[A-Z][a-z]+)", tail)
        return t[:m.start()] + ("\n\n" + tail[app.start():] if app else "")
    return t


def fetch(pid):
    html = subprocess.run(["curl", "-sL", "--max-time", "60", f"https://arxiv.org/html/{pid}v1"],
                          capture_output=True).stdout.decode("utf-8", "replace")
    if len(html) > 20_000 and "ltx_document" in html:
        t = html_to_text(html)
        if len(t) > MIN_HTML_CHARS:  # some HTML renders stop after the first table
            return t, f"https://arxiv.org/html/{pid}v1"
    pdf = OUT / f"{pid}.pdf"
    subprocess.run(["curl", "-sL", "--max-time", "90", "-o", str(pdf), f"https://arxiv.org/pdf/{pid}v1"])
    r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True)
    pdf.unlink(missing_ok=True)
    t = re.sub(r"[ \t]{2,}", " ", r.stdout.decode("utf-8", "replace"))
    return (t, f"https://arxiv.org/pdf/{pid}v1") if len(t) > 5_000 else (None, None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    papers = json.load(open(SEEN))["papers"]
    meta = {}
    for pid, p in sorted(papers.items()):
        if p["read"] != "full":
            continue
        dst = OUT / f"{pid}.txt"
        if dst.exists():
            meta[pid] = json.loads((OUT / f"{pid}.json").read_text())
            continue
        for attempt in range(2):
            t, src = fetch(pid)
            if t:
                break
            time.sleep(10)
        if not t:
            print("FAILED", pid, file=sys.stderr)
            continue
        raw = len(t)
        t = cut_refs(t)[:MAX_CHARS]
        dst.write_text(t)
        meta[pid] = {"source": src, "raw_chars": raw, "chars": len(t), "truncated": raw > MAX_CHARS}
        (OUT / f"{pid}.json").write_text(json.dumps(meta[pid]))
        print(pid, src.split("/")[3], raw, len(t))
        time.sleep(2)
    json.dump(meta, open(HERE / "cache" / "fetch_meta.json", "w"), indent=1)
    print(len(meta), "papers")


if __name__ == "__main__":
    main()
