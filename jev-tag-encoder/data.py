"""Build the three fixtures this experiment encodes.

  data/mixed.jsonl   step 2: 100 AG News test + 100 20 Newsgroups test (stratified)
  data/arxiv.jsonl   step 3/5: title+abstract, multi-label over LABELS (arXiv cross-lists)
  data/scifact/      step 4: BEIR SciFact (corpus, queries, test qrels)

Each record: {"id", "text", ...}. Deterministic given the upstream sources.
"""
import csv
import io
import json
import random
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_CHARS = 4000  # truncate long posts; 20NG has multi-page threads

# arXiv category -> the tag(s) in tags.txt it maps to. Label score = max over its tags.
LABELS = {
    "cs.LG": ["machine learning"],
    "cs.CL": ["large language models", "linguistics"],
    "cs.CV": ["computer vision"],
    "cs.CR": ["cryptography", "cybersecurity"],
    "cs.RO": ["robotics"],
    "cs.DB": ["databases"],
    "cs.NI": ["computer networking"],
    "cs.DC": ["distributed systems"],
    "cs.IR": ["search and information retrieval"],
    "cs.PL": ["programming languages"],
    "cs.OS": ["operating systems"],
    "quant-ph": ["quantum physics"],
    "astro-ph": ["astrophysics"],
    "hep": ["particle physics"],
    "cond-mat": ["condensed matter"],
    "cond-mat.mtrl-sci": ["materials science"],
    "physics.flu-dyn": ["fluid dynamics"],
    "physics.optics": ["optics"],
    "physics.ao-ph": ["meteorology", "climate science"],
    "physics.geo-ph": ["geology"],
    "q-bio.NC": ["neuroscience"],
    "q-bio.GN": ["genomics"],
    "q-bio.BM": ["protein structure", "biochemistry"],
    "q-bio.PE": ["evolution", "ecology"],
}
# arXiv search clause per label (families expand to their subcategories)
QUERY = {
    "astro-ph": "cat:astro-ph.GA OR cat:astro-ph.CO OR cat:astro-ph.SR OR cat:astro-ph.HE OR cat:astro-ph.EP",
    "hep": "cat:hep-ph OR cat:hep-ex",
    "cond-mat": "cat:cond-mat.str-el OR cat:cond-mat.supr-con OR cat:cond-mat.mes-hall",
}
PER_LABEL = 60


def label_of(cat: str) -> str | None:
    if cat in LABELS:
        return cat
    if cat.startswith("astro-ph"):
        return "astro-ph"
    if cat in ("hep-ph", "hep-ex"):
        return "hep"
    if cat in ("cond-mat.str-el", "cond-mat.supr-con", "cond-mat.mes-hall"):
        return "cond-mat"
    return None


def get(url: str, timeout: int = 120) -> bytes:
    for attempt in range(5):
        try:
            # urllib gets HTTP 406 from export.arxiv.org where curl and requests get 200
            r = requests.get(url, headers={"User-Agent": "muninn-raven"}, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001 — transient network, retried then raised
            if attempt == 4:
                raise
            print(f"retry {url[:80]}: {e}")
            time.sleep(3 * 2 ** attempt)


def build_mixed(rng: random.Random) -> None:
    out = DATA / "mixed.jsonl"
    if out.exists():
        return
    ag_names = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}
    rows = list(csv.reader(io.StringIO(get(
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv").decode())))
    by = {}
    for cls, title, desc in rows:
        by.setdefault(cls, []).append(f"{title}. {desc}".replace("\\", " "))
    recs = []
    for cls in sorted(by):
        for i, t in enumerate(rng.sample(by[cls], 25)):
            recs.append({"id": f"ag-{cls}-{i}", "source": "agnews", "label": ag_names[cls], "text": t})
    from sklearn.datasets import fetch_20newsgroups
    ng = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), data_home=str(DATA / "sk"))
    idx = [i for i, t in enumerate(ng.data) if len(t.strip()) >= 200]
    by_t = {}
    for i in idx:
        by_t.setdefault(int(ng.target[i]), []).append(i)
    picked = []
    for t in sorted(by_t):  # 5 per newsgroup = 100
        picked += rng.sample(by_t[t], 5)
    for i in picked:
        recs.append({"id": f"ng-{i}", "source": "20ng", "label": ng.target_names[ng.target[i]],
                     "text": ng.data[i].strip()[:MAX_CHARS]})
    DATA.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"mixed: {len(recs)}")


def build_arxiv(rng: random.Random) -> None:
    out = DATA / "arxiv.jsonl"
    if out.exists():
        return
    ns = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
    papers = {}
    for lab in LABELS:
        q = QUERY.get(lab, f"cat:{lab}")
        url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": 0, "max_results": PER_LABEL,
             "sortBy": "submittedDate", "sortOrder": "descending"}))
        root = ET.fromstring(get(url))
        n = 0
        for e in root.findall("a:entry", ns):
            aid = e.find("a:id", ns).text.rsplit("/", 1)[-1].split("v")[0]
            cats = [c.get("term") for c in e.findall("a:category", ns)]
            labs = sorted({label_of(c) for c in cats} - {None})
            title = " ".join(e.find("a:title", ns).text.split())
            abst = " ".join(e.find("a:summary", ns).text.split())
            if labs:
                papers[aid] = {"id": f"ax-{aid}", "cats": cats, "labels": labs, "query_label": lab,
                               "text": f"{title}. {abst}"[:MAX_CHARS]}
                n += 1
        print(f"arxiv {lab}: {n}")
        time.sleep(3.5)  # arXiv API etiquette: one request per 3 s
    recs = list(papers.values())
    rng.shuffle(recs)
    for i, r in enumerate(recs):
        r["split"] = "test" if i < 600 else "train"
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"arxiv: {len(recs)} papers, {sum(r['split'] == 'train' for r in recs)} train")


def build_scifact() -> None:
    d = DATA / "scifact"
    if (d / "corpus.jsonl").exists():
        return
    z = zipfile.ZipFile(io.BytesIO(get("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
                                       timeout=600)))
    d.mkdir(parents=True, exist_ok=True)
    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (d / Path(name).name).write_bytes(z.read(f"scifact/{name}"))
    print("scifact: ok")


if __name__ == "__main__":
    rng = random.Random(98)
    DATA.mkdir(exist_ok=True)
    build_mixed(rng)
    build_arxiv(rng)
    build_scifact()
