"""Colab L4 job: MedCPT vectors for the training texts and for every eval set (run via colab exec -f).

Inputs in /content: train.jsonl.gz (run/data.py), pubmed_eval.json. Outputs in /content/out:
teacher_train.f16.npy (row order = train.jsonl), and per eval set <name>_docs.f16.npy, <name>_queries.f16.npy.
Queries: MedCPT-Query-Encoder, max 64 tokens. Training passages: MedCPT-Article-Encoder on ["", passage],
max 128 (the student sees the passage alone). Eval docs: the article encoder on [title, text], max 512.
"""
import gzip, json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], check=False)
import numpy as np, pandas as pd, torch, csv
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

os.makedirs("/content/out", exist_ok=True)
dev = "cuda"
qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder"); qm = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(dev).half().eval()
at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder"); am = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(dev).half().eval()
LOG = open("/content/out/log.txt", "a")
def log(*a):
    print(*a, flush=True); LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()

@torch.inference_mode()
def enc(tok, model, items, max_len, bs, pair):
    order = np.argsort([len(x[0]) + len(x[1]) if pair else len(x) for x in items])
    out = np.zeros((len(items), 768), np.float16)
    t0 = time.time()
    for n, i in enumerate(range(0, len(order), bs)):
        ix = order[i:i + bs]
        batch = [list(items[j]) for j in ix] if pair else [items[j] for j in ix]
        f = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(dev)
        out[ix] = model(**f).last_hidden_state[:, 0].float().cpu().numpy().astype(np.float16)
        if n % 500 == 0:
            log(f"{i}/{len(items)} {time.time() - t0:.0f}s")
    return out

rows = [json.loads(l) for l in gzip.open("/content/train.jsonl.gz", "rt")]
qi = [i for i, r in enumerate(rows) if r["role"] == "q"]
di = [i for i, r in enumerate(rows) if r["role"] == "d"]
T = np.zeros((len(rows), 768), np.float16)
T[qi] = enc(qt, qm, [rows[i]["t"] for i in qi], 64, 512, False)
T[di] = enc(at, am, [("", rows[i]["t"]) for i in di], 128, 512, True)
np.save("/content/out/teacher_train.f16.npy", T)
log("train done", T.shape)

def beir(name):
    c = pd.read_parquet(hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    q = pd.read_parquet(hf_hub_download(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    keep = set()
    with open(hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset")) as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        keep = {qid for qid, _, s in r if int(s) > 0}
    q = q[q["_id"].astype(str).isin(keep)]
    return c["title"].fillna("").tolist(), c["text"].fillna("").tolist(), q["text"].tolist()

ev = json.load(open("/content/pubmed_eval.json"))
sets = {"pubmed": (ev["title"], ev["text"], ev["title"])}
for n in ["scifact", "nfcorpus", "trec-covid"]:
    sets[n] = beir(n)
for n, (ti, tx, qs) in sets.items():
    # pubmed eval: queries are titles, docs are abstracts WITHOUT their title
    docs = list(zip([""] * len(tx), tx)) if n == "pubmed" else list(zip(ti, tx))
    np.save(f"/content/out/{n}_docs.f16.npy", enc(at, am, docs, 512, 128, True))
    np.save(f"/content/out/{n}_queries.f16.npy", enc(qt, qm, qs, 64, 512, False))
    log(n, "done", len(docs), len(qs))
log("ALL DONE")
