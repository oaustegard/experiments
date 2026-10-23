"""Per-tag hub bias from a reference image set (CSLS, Lample et al. 2018).

Some tags ("dot", "twin", "cat furniture") sit close to many images and crowd every top-k.
CSLS ranks by cos(img, tag) - r_tag/2 ... we store r_tag = mean cosine of each tag to its
K nearest reference images; eval_coco.py subtracts BETA * r_tag. The reference set is ~1,000
random images from the Bluesky image-search index (thumbnails, not committed).

Writes .cache/tag_bias.f32.bin (N float32) and results/calibration.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_tags import pixels

HERE = Path(__file__).resolve().parents[1]
K = 10


def main():
    paths = sorted((HERE / ".cache/bsky").glob("*.jpg"))
    sess = ort.InferenceSession(str(HERE / ".cache/onnx/vision_model.onnx"))
    embs = []
    for p in paths:
        try:
            embs.append(sess.run(None, {"pixel_values": pixels(p)})[0][0])
        except Exception as e:
            print("skip", p.name, e, file=sys.stderr)
    E = np.stack(embs); E /= np.linalg.norm(E, axis=1, keepdims=True)
    np.save(HERE / ".cache/ref_embs.npy", E)
    bank = np.frombuffer((HERE / "web/tags.f16.bin").read_bytes(), dtype=np.float16).astype(np.float32).reshape(-1, 512)
    S = E @ bank.T                                  # images x tags
    r_tag = np.sort(S, axis=0)[-K:].mean(axis=0)    # mean of each tag's K nearest images
    (HERE / ".cache/tag_bias.f32.bin").write_bytes(r_tag.astype(np.float32).tobytes())
    tags = json.loads((HERE / "web/tags.json").read_text())["tags"]
    raw_top = np.argsort(-S, axis=1)[:, :5]
    cal_top = np.argsort(-(S - r_tag[None]), axis=1)[:, :5]
    def freq(top):
        c = np.bincount(top.ravel(), minlength=len(tags)); o = np.argsort(-c)[:15]
        return [[tags[i], int(c[i])] for i in o]
    out = {"n_images": len(E), "k": K,
           "distinct_tags_in_top5": {"raw": int(len(np.unique(raw_top))), "csls": int(len(np.unique(cal_top)))},
           "most_frequent_top5": {"raw": freq(raw_top), "csls": freq(cal_top)}}
    (HERE / "results/calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
