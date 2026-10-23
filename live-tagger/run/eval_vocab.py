"""Compare the small (RAM++) and big (RAM++ + Open Images V7) tag banks on the same 500 COCO images.

Same images and proxies as run/eval_coco.py (seed 1). Object recall@5 is reported twice:
over the COCO categories the small bank names verbatim (the same set for both banks, so the
numbers compare), and over each bank's own covered set. Also scores the big bank in float32
from the embedding cache, to price the int8 rows the page ships.

Writes results/eval_vocab.json.
"""
import json, random, re, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_tags import pixels

HERE = Path(__file__).resolve().parents[1]
C = HERE / ".cache"
N = 500


def words(s):
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", s.lower())}


def banks():
    small = json.loads((HERE / "web/tags.json").read_text())["tags"]
    big = json.loads((HERE / "web/tags-big.json").read_text())["tags"]
    raw = (HERE / "web/tags-big.i8.bin").read_bytes()
    q = np.frombuffer(raw[:len(big) * 512], dtype=np.int8).reshape(-1, 512).astype(np.float32)
    scale = np.frombuffer(raw[len(big) * 512:], dtype=np.float32)
    cache = dict(zip(json.loads((C / "tag_emb_cache.json").read_text()), np.load(C / "tag_emb_cache.npy")))
    return {
        "small_f16": (small, np.frombuffer((HERE / "web/tags.f16.bin").read_bytes(), dtype=np.float16).astype(np.float32).reshape(-1, 512)),
        "big_int8": (big, q * scale[:, None]),
        "big_f32": (big, np.stack([cache[t] for t in big])),
    }


def main():
    inst = json.loads((C / "annotations/instances_val2017.json").read_text())
    caps = json.loads((C / "annotations/captions_val2017.json").read_text())
    cat = {c["id"]: c["name"] for c in inst["categories"]}
    objs, capw = {}, {}
    for a in inst["annotations"]:
        objs.setdefault(a["image_id"], set()).add(cat[a["category_id"]])
    for a in caps["annotations"]:
        capw.setdefault(a["image_id"], set()).update(words(a["caption"]))
    ids = sorted(i for i in objs if i in capw)
    random.seed(1); ids = random.sample(ids, N)   # the same draw as eval_coco.py
    emb_path = C / "coco_embs.npy"
    if emb_path.exists():
        E = np.load(emb_path)
    else:
        sess = ort.InferenceSession(str(C / "onnx/vision_model.onnx"))
        E = np.stack([sess.run(None, {"pixel_values": pixels(C / "coco" / f"{i:012d}.jpg")})[0][0] for i in ids])
        E /= np.linalg.norm(E, axis=1, keepdims=True); np.save(emb_path, E)
    B = banks()
    shared = {o.lower() for o in cat.values()} & {t.lower() for t in B["small_f16"][0]}
    res, examples = {}, {}
    for name, (tags, bank) in B.items():
        tidx = {}
        for k, t in enumerate(tags):
            tidx.setdefault(t.lower(), k)
        tagw = [words(t) for t in tags]
        top = np.argsort(-(E @ bank.T), axis=1)[:, :5]
        rec_shared, rec_own, prec = [], [], []
        for r, i in enumerate(ids):
            gs = {tidx[o] for o in objs[i] if o in shared}
            go = {tidx[o] for o in objs[i] if o in tidx}
            if gs: rec_shared.append(len(gs & set(top[r])) / len(gs))
            if go: rec_own.append(len(go & set(top[r])) / len(go))
            prec.append(np.mean([tagw[t] <= capw[i] for t in top[r]]))
        res[name] = {"n_tags": len(tags),
                     "object_recall@5_shared": round(float(np.mean(rec_shared)), 3),
                     "object_recall@5_own": round(float(np.mean(rec_own)), 3),
                     "coco_categories_in_tags": sum(1 for o in set(cat.values()) if o in tidx),
                     "caption_precision@5": round(float(np.mean(prec)), 3),
                     "distinct_tags_in_top5": int(len(np.unique(top)))}
        examples[name] = {str(ids[r]): [tags[t] for t in top[r]] for r in range(8)}
        print(name, res[name])
    (HERE / "results/eval_vocab.json").write_text(json.dumps({"n_images": N, "shared_categories": len(shared),
                                                              "by_bank": res, "examples": examples}, indent=1))
    for k in examples["small_f16"]:
        print(k, "| small:", ", ".join(examples["small_f16"][k]), "| big:", ", ".join(examples["big_int8"][k]))


if __name__ == "__main__":
    main()
