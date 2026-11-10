"""Fetch the training set and write the 7-set zero-shot benchmark in GLiNER's word-level format.

    python3 run/prep_data.py [--zip PATH/IE_INSTRUCTIONS.zip] [--skip-train]

(a) urchade/pile-mistral-v0.1 data.json -> data/train/pile_mistral.json (19,724 passages).
(b) IE_INSTRUCTIONS.zip (InstructUIE; --zip, else data/IE_INSTRUCTIONS.zip, else downloaded from
    Google Drive and sha256-checked) -> data/bench/<set>/{test,dev,labels}.json for the seven
    zero-shot sets only. Records are converted by the installed library's own
    gliner.evaluation.evaluate_ner.process (sentence.split(), word indices, end inclusive, type
    lowercased) and labels are lowercased as create_dataset does, so the files are exactly what
    upstream's evaluator would score. An entity whose characters do not align with the split
    words keeps upstream's (None, None, type) form (null in JSON); upstream counts it as a gold
    entity that no prediction can match, and so do we. dev is written only for the five CrossNER
    sets (the MIT dev files are byte-identical to test). Train splits are never read or written.

Counts are checked against prior-art.md section 3.1; a mismatch exits 1.
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from gliner.evaluation.evaluate_ner import process

D = Path(__file__).resolve().parents[1]
ZIP_URL = ("https://drive.usercontent.google.com/download?id=1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT"
           "&export=download&confirm=t")
ZIP_SHA = "a3308ef7fc26842b9ff5819eb7e3c8e05511a8bebb6e7d148aeec005e0413a58"
CROSSNER = ["CrossNER_AI", "CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science"]
MIT = ["mit-movie", "mit-restaurant"]
SETS = CROSSNER + MIT
# (sentences, entity mentions) from prior-art.md 3.1, measured on the extracted files
EXPECT = {
    ("CrossNER_AI", "dev"): (350, 1546), ("CrossNER_AI", "test"): (431, 1804),
    ("CrossNER_literature", "dev"): (400, 2125), ("CrossNER_literature", "test"): (416, 2263),
    ("CrossNER_music", "dev"): (380, 2679), ("CrossNER_music", "test"): (465, 3334),
    ("CrossNER_politics", "dev"): (540, 3472), ("CrossNER_politics", "test"): (650, 4200),
    ("CrossNER_science", "dev"): (450, 2537), ("CrossNER_science", "test"): (543, 3088),
    ("mit-movie", "test"): (2442, 5338), ("mit-restaurant", "test"): (1520, 3150),
}
LABELS = {"CrossNER_AI": 14, "CrossNER_literature": 12, "CrossNER_music": 13, "CrossNER_politics": 9,
          "CrossNER_science": 17, "mit-movie": 12, "mit-restaurant": 8}
TRAIN_N = 19724


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_zip(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    print(f"downloading IE_INSTRUCTIONS.zip -> {dest}", flush=True)
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.rename(dest)


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.rename(p)


def prep_train():
    from huggingface_hub import hf_hub_download
    out = D / "data/train/pile_mistral.json"
    if not out.exists():
        src = hf_hub_download("urchade/pile-mistral-v0.1", "data.json", repo_type="dataset")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out.with_suffix(".tmp"))
        out.with_suffix(".tmp").rename(out)
    data = json.loads(out.read_text())
    n_ment = sum(len(x["ner"]) for x in data)
    n_empty = sum(1 for x in data if not x["ner"])
    print(f"train pile_mistral: {len(data)} passages, {n_ment} mentions, {n_empty} with no entity, "
          f"{len({t.lower() for x in data for *_, t in x['ner']})} distinct lowercased types")
    return len(data) == TRAIN_N


def prep_bench(zpath):
    if sha256(zpath) != ZIP_SHA:
        print(f"sha256 mismatch for {zpath}", file=sys.stderr)
        return False
    ok = True
    with zipfile.ZipFile(zpath) as z:
        for s in SETS:
            labels = json.loads(z.read(f"NER/{s}/labels.json"))
            labels = [x.lower() for x in labels]
            write_json(D / f"data/bench/{s}/labels.json", labels)
            line = [f"{s:20} labels {len(labels):2}"]
            ok &= len(labels) == LABELS[s]
            for split in (["dev", "test"] if s in CROSSNER else ["test"]):
                raw = json.loads(z.read(f"NER/{s}/{split}.json"))
                conv = [process(r) for r in raw]
                write_json(D / f"data/bench/{s}/{split}.json", conv)
                n_sent, n_ent = len(conv), sum(len(r["ner"]) for r in conv)
                unal = sum(1 for r in conv for e in r["ner"] if e[0] is None or e[1] is None)
                exp = EXPECT[(s, split)]
                flag = "ok" if (n_sent, n_ent) == exp else f"MISMATCH expected {exp}"
                ok &= (n_sent, n_ent) == exp
                line.append(f"{split} {n_sent:5} sent {n_ent:5} ents ({unal} unaligned) {flag}")
            print(" | ".join(line))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    ok = True
    if not a.skip_train:
        ok &= prep_train()
    zpath = a.zip or (D / "data/IE_INSTRUCTIONS.zip")
    if not zpath.exists():
        fetch_zip(zpath)
    ok &= prep_bench(zpath)
    print("PREP_OK" if ok else "PREP_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
