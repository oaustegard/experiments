"""Title, authors, date and abstract for every ranked paper.

    python3 fetch_meta.py      # data/papers_ranked.json -> data/papers.json

arXiv ids go to the arXiv API; DOIs to Crossref, then Semantic Scholar, then the landing page's
citation_/dc./og: meta tags. A paper whose abstract cannot be found by any route is kept with
abstract null and excluded from training (reported, not hidden).
"""
import html, json, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "muninn-experiments/1.0 (mailto:oskar@austegard.com)"}


def get(url, accept=None):
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    for t in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30).read()
        except Exception as e:
            if getattr(e, "code", None) in (404, 400):
                return None
            time.sleep(2 * (t + 1))
    return None


def clean(s):
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    s = re.sub(r"^\s*(abstract|summary)\s*[:.]?\s*", "", re.sub(r"\s+", " ", s).strip(), flags=re.I)
    return s or None


def arxiv(ids):
    raw = get("https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for e in ET.fromstring(raw).findall("a:entry", ns):
        aid = re.search(r"abs/(\d{4}\.\d{4,5})", e.find("a:id", ns).text).group(1)
        out[aid] = {"title": clean(e.find("a:title", ns).text), "abstract": clean(e.find("a:summary", ns).text),
                    "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                    "date": e.find("a:published", ns).text[:10], "venue": "arXiv", "abstract_src": "arxiv"}
    return out


def crossref(doi):
    raw = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not raw:
        return None
    m = json.loads(raw)["message"]
    d = (m.get("published") or m.get("issued") or {}).get("date-parts", [[None]])[0]
    return {"title": clean((m.get("title") or [None])[0]), "abstract": clean(m.get("abstract")),
            "authors": [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in m.get("author", [])],
            "date": "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(d) if x) if d and d[0] else None,
            "venue": (m.get("container-title") or [m.get("publisher")])[0], "abstract_src": "crossref"}


def s2(doi):
    raw = get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=title,abstract,authors,venue,publicationDate")
    if not raw:
        return None
    m = json.loads(raw)
    return {"title": m.get("title"), "abstract": clean(m.get("abstract")), "authors": [a["name"] for a in m.get("authors", [])],
            "date": m.get("publicationDate"), "venue": m.get("venue"), "abstract_src": "semanticscholar"}


def landing(doi):
    raw = get("https://doi.org/" + doi, accept="text/html")
    if not raw:
        return None
    t = raw.decode("utf-8", "ignore")
    for name in ("citation_abstract", "dc.description", "DC.Description", "description", "og:description"):
        m = re.search(r'<meta[^>]+(?:name|property)="%s"[^>]+content="([^"]+)"' % re.escape(name), t, re.I) or \
            re.search(r'<meta[^>]+content="([^"]+)"[^>]+(?:name|property)="%s"' % re.escape(name), t, re.I)
        if m and len(clean(m.group(1)) or "") > 300:
            return {"abstract": clean(m.group(1)), "abstract_src": "landing:" + name}
    return None


def main():
    ranked = json.loads((HERE / "data/papers_ranked.json").read_text())
    ax = arxiv([r["id"][6:] for r in ranked if r["id"].startswith("arxiv:")])
    out = []
    for i, r in enumerate(ranked):
        if r["id"].startswith("arxiv:"):
            m = ax.get(r["id"][6:])
        else:
            doi = r["id"][4:]
            m = crossref(doi) or {}
            if not m.get("abstract"):
                s = s2(doi) or {}
                m = {**s, **{k: v for k, v in m.items() if v}} if s.get("abstract") else m
                if s.get("abstract"):
                    m["abstract"], m["abstract_src"] = s["abstract"], "semanticscholar"
            if not m.get("abstract"):
                m.update(landing(doi) or {})
            time.sleep(1.1)
        m = m or {}
        out.append({"id": r["id"], "rank": i + 1, "engagement": r["engagement"], "posts": r["posts"],
                    "title": m.get("title"), "authors": m.get("authors"), "date": m.get("date"), "venue": m.get("venue"),
                    "abstract": m.get("abstract") if m.get("abstract") and len(m["abstract"]) >= 300 else None,
                    "abstract_src": m.get("abstract_src"), "post_text": r["texts"][0]})
        print(i + 1, r["id"], (m.get("abstract_src") if out[-1]["abstract"] else "NO ABSTRACT"), (m.get("title") or "")[:60], flush=True)
    (HERE / "data/papers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(sum(bool(x["abstract"]) for x in out), "of", len(out), "with abstracts")


if __name__ == "__main__":
    main()
