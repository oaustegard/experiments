"""Reference tags for test images: MobileCLIP-S0 vision_model.onnx (fp32) + the tag bank.

Preprocessing matches the page: center square crop, resize to 256 (bilinear), /255, CHW,
no mean/std normalization (preprocessor_config.json: do_normalize false).
Writes results/ref_tags.json: per image, the 512-d embedding and the top 10 tags.
"""
import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / ".cache"


def pixels(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def main(paths):
    sess = ort.InferenceSession(str(CACHE / "onnx/vision_model.onnx"))
    bank = np.frombuffer((HERE / "web/tags.f16.bin").read_bytes(), dtype=np.float16).astype(np.float32).reshape(-1, 512)
    tags = json.loads((HERE / "web/tags.json").read_text())["tags"]
    out = {}
    for p in paths:
        v = sess.run(None, {"pixel_values": pixels(p)})[0][0]
        v = v / np.linalg.norm(v)
        s = bank @ v
        top = np.argsort(-s)[:10]
        out[Path(p).name] = {"top": [[tags[i], round(float(s[i]), 4)] for i in top], "emb": [round(float(x), 6) for x in v]}
        print(Path(p).name, ", ".join(f"{tags[i]} {s[i]:.3f}" for i in top[:8]))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/ref_tags.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
