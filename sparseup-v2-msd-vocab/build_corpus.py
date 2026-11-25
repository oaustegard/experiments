"""Training corpus: PubMed abstracts that name each MSD catalog concept, fetched from NCBI E-utilities.

    python3 build_corpus.py      # -> cache/train_pubmed.txt ('pmid ||| title ||| mesh;... ||| abstract'), data/train_pmids.tsv

Per concept, one esearch for its catalog names and HGNC symbol in title/abstract (English, with abstract,
2010 on), up to 80 PMIDs. Every PMID of the evaluation fixture (../sparseup-v2-msd-probe/tag-vector/data/seed_pmids.tsv)
is excluded, so no evaluation abstract is trained on.
"""
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent / "sparseup-v2-msd-probe" / "tag-vector"
UA = {"User-Agent": "experiments-msd-vocab/1"}
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda", "ω": "omega"}


def eutil(name, params):
    for k in range(5):
        try:
            req = urllib.request.Request(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{name}.fcgi",
                                         data=urllib.parse.urlencode(params).encode(), headers=UA)
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:  # noqa: BLE001  (429 / transient)
            time.sleep(2 + 3 * k)
            err = e
    raise err


def terms(c):
    out = []
    for n in c["catalog_names"]:
        for t in [n, re.sub(r"\s*\([^)]*\)", "", n)] + n.split("/"):
            t = t.strip()
            for g, s in GREEK.items():
                t = t.replace(g, s)
            if len(t) >= 2 and t not in out:
                out.append(t)
    if c.get("symbol") and c["symbol"] not in out:
        out.append(c["symbol"])
    return out[:6]


def main():
    vocab = json.loads((TV / "data" / "vocab.json").read_text())
    excl = {p for line in (TV / "data" / "seed_pmids.tsv").read_text().splitlines() if line.strip()
            for p in line.split("\t")[1].split()}
    per = {}
    for i, (cid, c) in enumerate(vocab["concepts"].items()):
        q = "(" + " OR ".join(f'"{t}"[tiab]' for t in terms(c)) + ') AND hasabstract AND english[la] AND 2010:2026[dp]'
        ids = json.loads(eutil("esearch", {"db": "pubmed", "term": q, "retmax": 80, "retmode": "json", "sort": "relevance"}))
        per[cid] = [p for p in ids["esearchresult"].get("idlist", []) if p not in excl]
        time.sleep(0.35)
        if i % 100 == 0:
            print("search", i, sum(map(len, per.values())), flush=True)
    (HERE / "data" / "train_pmids.tsv").write_text("".join(f"{cid}\t{' '.join(p)}\n" for cid, p in per.items()))
    pmids = sorted({p for v in per.values() for p in v}, key=int)
    out = []
    for i in range(0, len(pmids), 200):
        root = ET.fromstring(eutil("efetch", {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids[i:i + 200])}))
        for a in root.iter("PubmedArticle"):
            pmid = a.findtext(".//MedlineCitation/PMID")
            at = a.find(".//ArticleTitle")
            title = "".join(at.itertext()).strip() if at is not None else ""
            parts = []
            for t in a.findall(".//Abstract/AbstractText"):
                txt = "".join(t.itertext()).strip()
                parts.append(f"{t.get('Label')}: {txt}" if t.get("Label") else txt)
            mesh = [d.text for d in a.findall(".//MeshHeadingList/MeshHeading/DescriptorName")]
            abstract = " ".join(parts).replace("\n", " ")
            if len(abstract) >= 200:
                out.append(" ||| ".join([pmid, title.replace("|", "/"), ";".join(mesh), abstract.replace("|||", "/")]))
        if i % 4000 == 0:
            print("fetch", i, len(out), flush=True)
        time.sleep(0.35)
    (HERE / "cache" / "train_pubmed.txt").write_text("\n".join(out) + "\n")
    print("concepts", len(per), "pmids", len(pmids), "abstracts", len(out))


if __name__ == "__main__":
    main()
