"""Build kaggle/job_<name>.py: the whole stage 0/1/2 run on Kaggle's 2x T4, rebuilding every input on the box.

    python3 kaggle_job.py --name run1 [-- train.py args, e.g. --stages 0,1 --steps1 1200]
    python3 <claude-workspace>/scripts/kaggle_job.py push sparseup-msd-run1 kaggle/job_run1.py --accel gpu --internet

On the box: HGNC from its public bucket (sha256 checked) -> build_vocab.py -> vocab.json; the fixture's seed PMIDs
and the training PMIDs from NCBI E-utilities -> build_fixture.py with concepts and split pinned to concepts.tsv;
SciFact from the Hub; the model from the Hub; then train.py. Outputs (sparse encodes, the trained rows or model)
land in /kaggle/working/out. Nothing here is uploaded anywhere else.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent / "sparseup-v2-msd-probe" / "tag-vector"
FILES = {
    "data/msd_analytes.txt": HERE.parent / "sparseup-v2-msd-probe" / "data" / "msd_analytes.txt",
    "tv/tagger.py": TV / "tagger.py", "tv/build_vocab.py": TV / "build_vocab.py", "tv/build_fixture.py": TV / "build_fixture.py",
    "tv/data/concepts.tsv": TV / "data" / "concepts.tsv", "tv/data/mesh_map.json": TV / "data" / "mesh_map.json",
    "tv/data/seed_pmids.tsv": TV / "data" / "seed_pmids.tsv",
    "mv/msd_vocab.py": HERE / "msd_vocab.py", "mv/train.py": HERE / "train.py", "mv/train_pmids.tsv": HERE / "data" / "train_pmids.tsv",
}

RUNNER = r'''
import hashlib, json, os, shutil, subprocess, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
W = Path("/kaggle/working"); R = W / "work"; OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
for rel, src in FILES.items():
    (R / rel).parent.mkdir(parents=True, exist_ok=True); (R / rel).write_text(src)
def sh(cmd, cwd=None, env=None):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if r.returncode:
        raise SystemExit(f"failed: {cmd}")
def eutil(name, params):
    for k in range(6):
        try:
            req = urllib.request.Request(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{name}.fcgi",
                                         data=urllib.parse.urlencode(params).encode(), headers={"User-Agent": "experiments-msd-vocab/1"})
            return urllib.request.urlopen(req, timeout=180).read()
        except Exception as e:
            print("eutils retry", k, e, flush=True); time.sleep(3 + 5 * k)
    raise SystemExit("eutils failed")
def fetch(pmids, path, min_abs=0):
    out = []
    for i in range(0, len(pmids), 200):
        root = ET.fromstring(eutil("efetch", {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids[i:i + 200])}))
        for a in root.iter("PubmedArticle"):
            pmid = a.findtext(".//MedlineCitation/PMID"); at = a.find(".//ArticleTitle")
            title = "".join(at.itertext()).strip() if at is not None else ""
            parts = []
            for t in a.findall(".//Abstract/AbstractText"):
                txt = "".join(t.itertext()).strip(); parts.append(f"{t.get('Label')}: {txt}" if t.get("Label") else txt)
            abstract = " ".join(parts).replace("\n", " ")
            mesh = [d.text for d in a.findall(".//MeshHeadingList/MeshHeading/DescriptorName")]
            if len(abstract) >= min_abs:
                out.append(" ||| ".join([pmid, title.replace("|", "/"), ";".join(mesh), (abstract or "NA").replace("|||", "/")]))
        time.sleep(0.34)
    Path(path).write_text("\n".join(out) + "\n"); print("fetched", path, len(out), flush=True)
# 1. vocabulary
urllib.request.urlretrieve("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt", W / "hgnc.txt")
assert hashlib.sha256((W / "hgnc.txt").read_bytes()).hexdigest() == "46b83736ec351f15e22331575dce2f08416c5233765b26a08c65e6b42e56b651", "HGNC changed"
sh(f"{sys.executable} build_vocab.py {W / 'hgnc.txt'} > /dev/null", cwd=R / "tv")
# 2. fixture (all seed PMIDs; concepts and split pinned to the published concepts.tsv)
seed = sorted({p for l in (R / "tv/data/seed_pmids.tsv").read_text().splitlines() if l.strip() for p in l.split("\t")[1].split()}, key=int)
fetch(seed, W / "seed_pubmed.txt")
sh(f"{sys.executable} build_fixture.py {W / 'seed_pubmed.txt'}", cwd=R / "tv")
fx = json.loads((R / "tv/data/fixture.json").read_text())
lines = (R / "tv/data/concepts.tsv").read_text().splitlines(); hdr = lines[0].split("\t")
pub = {r[0]: dict(zip(hdr, r)) for r in (l.split("\t") for l in lines[1:])}
keep = []
for c in fx["concepts"]:
    p = pub.get(str(c["idx"]))
    if p:
        c["split"] = p["split"]; keep.append(c)
fx["concepts"] = keep
print("fixture docs", len(fx["docs"]), "concepts", len(keep), flush=True)
D = R / "mv" / "data"; D.mkdir(exist_ok=True)
(D / "fixture.json").write_text(json.dumps(fx, ensure_ascii=False))
for f in ("vocab.json",):
    shutil.copy(R / "tv/data" / f, D / f)
for f in ("concepts.tsv", "mesh_map.json"):
    shutil.copy(R / "tv/data" / f, D / f)
shutil.copy(R / "mv/train_pmids.tsv", D / "train_pmids.tsv")
# 3. training corpus
tp = sorted({p for l in (D / "train_pmids.tsv").read_text().splitlines() if l.strip() for p in l.split("\t")[1].split()}, key=int)
fetch(tp, D / "train_pubmed.txt", min_abs=200)
# 4. SciFact + model
from huggingface_hub import hf_hub_download
import pandas as pd
sfd = W / "scifact"
for repo, f in [("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet"), ("BeIR/scifact", "queries/queries-00000-of-00001.parquet"), ("BeIR/scifact-qrels", "test.tsv")]:
    hf_hub_download(repo, f, repo_type="dataset", local_dir=sfd)
corpus = pd.read_parquet(sfd / "corpus/corpus-00000-of-00001.parquet"); queries = pd.read_parquet(sfd / "queries/queries-00000-of-00001.parquet")
qrels = pd.read_csv(sfd / "test.tsv", sep="\t", dtype=str)
qids = list(dict.fromkeys(qrels["query-id"]))
qtext = dict(zip(queries["_id"].astype(str), queries["text"]))
sf = {"doc_ids": corpus["_id"].astype(str).tolist(), "docs": ((corpus["title"].fillna("") + " " + corpus["text"].fillna("")).str.strip()).tolist(),
      "query_ids": qids, "queries": [qtext[q] for q in qids]}
(D / "scifact.json").write_text(json.dumps(sf)); (OUT / "scifact_ids.json").write_text(json.dumps({"doc_ids": sf["doc_ids"], "query_ids": qids}))
for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
    hf_hub_download("Linkup-Platform/linkup-sparseup-embed-v2", f, local_dir=R / "mv/hf")
import torch; print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
import transformers; print("transformers", transformers.__version__, flush=True)
# 5. train + encode
env = dict(os.environ, PYTHONPATH=str(R / "tv"))
rc = subprocess.run(f"{sys.executable} -u train.py --hf hf --data data --out {OUT} {ARGS}", shell=True, cwd=R / "mv", env=env).returncode
print("train.py exit", rc, flush=True)   # non-fatal: a later stage failing must not discard earlier outputs
print("done", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a, rest = ap.parse_known_args()
    args = " ".join(x for x in rest if x != "--")
    text = {k: Path(v).read_text() for k, v in FILES.items()}
    out = HERE / "kaggle" / f"job_{a.name}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"FILES = {text!r}\nARGS = {args!r}\n" + RUNNER)
    print(out, out.stat().st_size, file=sys.stderr)


if __name__ == "__main__":
    main()
