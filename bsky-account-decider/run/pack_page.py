"""Pack the trained account decider for the page and score the packed build.

    python3 -I run/pack_page.py [CKPT_NAME]      # default all-s0

Uses browser-decider's packer (int8 matmuls in ±63, 4-bit token table, f32 head) and its NumPy
reference, so the numbers here are what the browser computes. Writes web/models/account/
(weights.bin, manifest.json, tokenizer.json, questions.json, post_question.json, about.json) and
results/packed_<CKPT_NAME>.json (accuracy against Sonnet per question on the 300 test accounts).
"""
import json
import shutil
import sys
from pathlib import Path

import brotli
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BD = ROOT.parent / "browser-decider"
sys.path.insert(0, str(BD / "run"))
sys.path.insert(0, str(ROOT / "run"))
from common import examples, score_examples  # noqa: E402
from pack import Ref, pack  # noqa: E402
from train_enc import Enc  # noqa: E402
from build_data import POST_Q  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "all-s0"
    score = "--no-score" not in sys.argv
    ck = ROOT / "out" / "ckpt" / name
    meta = json.loads((ck / "meta.json").read_text())
    from huggingface_hub import snapshot_download
    init = snapshot_download("jhu-clsp/ettin-encoder-17m", allow_patterns=["*.json", "*.bin", "*.safetensors"])
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(init).to_dict()
    sd = {k: v.numpy() for k, v in torch.load(ck / "best.pt", map_location="cpu").items()}
    man, data = pack(sd, cfg, "t4-w8L63", meta["temps"])
    dst = ROOT / "web" / "models" / "account"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "weights.bin").write_bytes(data)
    (dst / "manifest.json").write_text(json.dumps(man))
    shutil.copy(Path(init) / "tokenizer.json", dst / "tokenizer.json")
    shutil.copy(ROOT / "run" / "questions.json", dst / "questions.json")
    (dst / "post_question.json").write_text(json.dumps(POST_Q, indent=1))

    ref, tok = Ref(man, data), Enc(Path(init) / "tokenizer.json")
    by, overall = {}, None
    recs = [json.loads(l) for l in (ROOT / "data" / "cases.jsonl").open()]
    te = examples([r for r in recs if r["split"] == "test"]) if score else []
    probs = []
    for e in te:
        ids, pos = tok.build(e)
        z = ref.logits(ids, pos) / man["temps"][e["type"]]
        z = np.exp(z - z.max())
        probs.append((z / z.sum()).tolist())
    by = {}
    for q in sorted({e["qname"] for e in te}):
        sel = [i for i, e in enumerate(te) if e["qname"] == q]
        by[q] = score_examples([te[i] for i in sel], [probs[i] for i in sel])[0]
    if score:
        overall, _ = score_examples(te, probs)
        res = {"ckpt": name, "variant": "t4-w8L63", "overall": overall, "by_question": by}
        (ROOT / "results" / f"packed_{name}.json").write_text(json.dumps(res, indent=1))
    sizes = {f: len(brotli.compress((dst / f).read_bytes(), quality=11)) for f in ("weights.bin", "tokenizer.json", "manifest.json")}
    js = sum(len(brotli.compress(p.read_bytes(), quality=11)) for p in
             [BD / "web" / "decider.js", BD / "web" / "engine.js", BD / "web" / "wgsl" / "runtime.js",
              BD / "web" / "wgsl" / "kernels.js", ROOT / "web" / "state.js"])
    about = {"ckpt": name, "downloadMB": (sum(sizes.values()) + js) / 1e6, "brotliBytes": sizes, "jsBrotliBytes": js,
             "testAccuracy": {q: round(v["accuracy"], 4) for q, v in by.items()},
             "examples": ["austegard.com", "bsky.app", "nytimes.com"]}
    (dst / "about.json").write_text(json.dumps(about, indent=1))
    print(json.dumps({"overall": overall and overall["accuracy"], "by": about["testAccuracy"], "MB": about["downloadMB"]}, indent=1))


if __name__ == "__main__":
    main()
