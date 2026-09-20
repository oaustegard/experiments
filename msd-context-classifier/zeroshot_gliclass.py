"""Zero-shot floor: knowledgator/gliclass-modern-base-v2.0 over the same test split and queries.
Label text = taxonomy.json[label]["description"] if present, else the label name with '-' and '/' -> ' '.
Writes results/gliclass_zeroshot.json. Completion line: 'done -> <path>'."""
import json, os, sys, time
import numpy as np
from transformers import AutoTokenizer
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, split_rows, texts_of, evaluate, log

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "gliclass-modern-base-v2.0")
corpus = os.path.join(HERE, "data", "corpus.jsonl"); qpath = os.path.join(HERE, "data", "queries.jsonl")
rows = load_corpus(corpus); labels = sorted({r["label"] for r in rows}); lid = {l: i for i, l in enumerate(labels)}
tax = {}
tp = os.path.join(HERE, "data", "label_descriptions.json")  # {label: "short natural-language description"}
if os.path.exists(tp):
    tax = json.load(open(tp))
label_text = [tax.get(l) or l.replace("-", " ").replace("/", " ").replace("_", " ") for l in labels]
log("labels ->", dict(zip(labels, label_text)))
tr, dv, te = split_rows(rows)
tok = AutoTokenizer.from_pretrained(MODEL); model = GLiClassModel.from_pretrained(MODEL)
pipe = ZeroShotClassificationPipeline(model, tok, classification_type="single-label", device="cpu", max_length=512)


def score(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        res = pipe([t[:3000] for t in texts[i:i + bs]], label_text, threshold=0.0, batch_size=bs)
        for r in res:
            s = {x["label"]: x["score"] for x in r}
            out.append([s.get(lt, 0.0) for lt in label_text])
        if (i // bs) % 10 == 0: log(f"scored {min(i+bs, len(texts))}/{len(texts)}")
    p = np.clip(np.array(out), 1e-6, 1 - 1e-6)
    return np.log(p)  # logits-like for the shared evaluate()


t0 = time.time()
ldv = score(texts_of(dv)); lte = score(texts_of(te))
ydv = np.array([lid[r["label"]] for r in dv]); yte = np.array([lid[r["label"]] for r in te])
ts = []
for x in texts_of(te)[:10]:
    t = time.perf_counter(); score([x], 1); ts.append(time.perf_counter() - t)
res = evaluate("gliclass", ldv, ydv, lte, yte, len(labels), labels,
               {"arm": "zeroshot", "model": MODEL, "label_text": label_text, "ms_per_example_bs1_torch": 1000 * float(np.median(ts)), "wall_seconds": time.time() - t0})
if os.path.exists(qpath):
    qs = [json.loads(l) for l in open(qpath) if l.strip()]; qs = [q for q in qs if q["label"] in lid]
    lq = score([q["text"] for q in qs]); yq = np.array([lid[q["label"]] for q in qs])
    res["queries"] = evaluate("gliclass", ldv, ydv, lq, yq, len(labels), labels)
res["labels"] = labels
out = os.path.join(HERE, "results", "gliclass_zeroshot.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
log(f"acc {res['accuracy']:.3f} macroF1 {res['macro_f1']:.3f}" + (f" | queries acc {res['queries']['accuracy']:.3f}" if "queries" in res else ""))
print(f"done -> {out}", flush=True)
