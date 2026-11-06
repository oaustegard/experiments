"""Fetch PubMed abstracts (2023-2025, broad life-science query) via E-utilities."""
import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
import sys
YR = sys.argv[1] if len(sys.argv) > 1 else "2023:2025"
OUT = Path(__file__).resolve().parent / "data" / f"abstracts_{YR.replace(':','-')}.jsonl"
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            time.sleep(2 ** i)
    raise RuntimeError(url)
q = ("(protein[tiab] OR gene[tiab] OR cytokine[tiab] OR antibody[tiab] OR assay[tiab] OR "
     "compound[tiab] OR inhibitor[tiab] OR receptor[tiab]) AND hasabstract AND " + YR + "[dp]")
r = json.loads(get(E + "esearch.fcgi?" + urllib.parse.urlencode(
    {"db": "pubmed", "term": q, "retmax": 20000, "retmode": "json", "sort": "pub_date"})))
ids = r["esearchresult"]["idlist"]
print("ids", len(ids), "of", r["esearchresult"]["count"])
n = 0
with OUT.open("w") as f:
    for i in range(0, len(ids), 200):
        x = get(E + "efetch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(ids[i:i+200]), "retmode": "xml"}))
        for art in ET.fromstring(x).iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            t = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            a = " ".join("".join(e.itertext()) for e in art.iter("AbstractText"))
            if len(a) > 400:
                f.write(json.dumps({"pmid": pmid, "text": (t + " " + a).strip()}) + "\n"); n += 1
        time.sleep(0.4)
print("kept", n)
