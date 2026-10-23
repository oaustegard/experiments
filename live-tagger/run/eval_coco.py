"""Score the tag bank on 500 COCO val2017 images, with and without the hub bias.

Two proxies, since COCO has no tag ground truth:
- object recall@5: of the image's COCO categories whose name is a tag, the share in the top 5;
- caption precision@5: share of the top-5 tags whose words all appear in the image's 5 captions
  (simple lowercase match with a trailing-s strip).
BETA scales the bias subtracted: score = cos - BETA * bias. `--bias csls` (default) uses
run/calibrate.py's r_tag; `--bias mean` uses each tag's mean cosine over the same Bluesky
reference images (.cache/ref_embs.npy) and writes results/eval_coco_meanbias.json.
"""
import json, random, re, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_tags import pixels

HERE = Path(__file__).resolve().parents[1]
C = HERE / ".cache"
N, BETAS = 500, [0, 0.25, 0.5, 0.75, 1.0]


def words(s):
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", s.lower())}


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
    random.seed(1); ids = random.sample(ids, N)
    (C / "coco").mkdir(exist_ok=True)
    def get(i):
        p = C / "coco" / f"{i:012d}.jpg"
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", timeout=30).read())
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(get, ids))
    sess = ort.InferenceSession(str(C / "onnx/vision_model.onnx"))
    E = np.stack([sess.run(None, {"pixel_values": pixels(C / "coco" / f"{i:012d}.jpg")})[0][0] for i in ids])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    bank = np.frombuffer((HERE / "web/tags.f16.bin").read_bytes(), dtype=np.float16).astype(np.float32).reshape(-1, 512)
    if MODE == "mean":
        bias = (np.load(C / "ref_embs.npy") @ bank.T).mean(axis=0)
    else:
        bias = np.frombuffer((C / "tag_bias.f32.bin").read_bytes(), dtype=np.float32)
    tags = json.loads((HERE / "web/tags.json").read_text())["tags"]
    tidx = {t.lower(): k for k, t in enumerate(tags)}
    tagw = [words(t) for t in tags]
    S = E @ bank.T
    res = {}
    for b in BETAS:
        top = np.argsort(-(S - b * bias[None]), axis=1)[:, :5]
        rec, prec = [], []
        for r, i in enumerate(ids):
            gt = {tidx[o] for o in objs[i] if o in tidx}
            if gt:
                rec.append(len(gt & set(top[r])) / len(gt))
            prec.append(np.mean([tagw[t] <= capw[i] for t in top[r]]))
        res[str(b)] = {"object_recall@5": round(float(np.mean(rec)), 3), "caption_precision@5": round(float(np.mean(prec)), 3),
                       "n_recall": len(rec), "distinct_tags": int(len(np.unique(top)))}
        print(b, res[str(b)])
    coverage = sum(1 for n in set(cat.values()) if n in tidx)
    (HERE / ("results/eval_coco_meanbias.json" if MODE == "mean" else "results/eval_coco.json")).write_text(json.dumps({"n_images": N, "coco_categories_in_tags": coverage, "by_beta": res}, indent=1))


if __name__ == "__main__":
    MODE = sys.argv[sys.argv.index("--bias") + 1] if "--bias" in sys.argv else "csls"
    main()
