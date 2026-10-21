"""Per-query results for the embed-coord board (experiments#104), ASYM metric: query encoder
varies, docs are EG2 q4's committed 2-bit codes.

Usage: python3 -I run/perquery.py [--tags l2_mac_k2 l2_mac_k4 l2_mac_k8]

Writes results/<arm>_perquery.json for EG2 (the teacher targets) and each shipped q4 build:
synth_hit10 = 1,498 bools in synth_test file order (source post in top 10), and
real_overlap10 = 1,000 floats in real_test file order (share of EG2's top 10 returned).
"""
import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import Bench  # noqa: E402
from export import ort_embed  # noqa: E402
from model import ROOT, tokenize  # noqa: E402
from common import Teacher, overlap  # noqa: E402


def dump(arm, B, top_synth, top_real, extra):
    hit = (top_synth == B.src[:, None]).any(1)
    out = {"arm": arm, "metric": "ASYM (docs = EG2 q4 committed 2-bit codes)",
           "synth_hit10": [bool(h) for h in hit],
           "real_overlap10": [round(float(x), 1) for x in overlap(top_real, B.top["real_test"])],
           "hit10_mean": float(hit.mean()), **extra}
    (ROOT / "results" / f"{arm}_perquery.json").write_text(json.dumps(out))
    print(arm, round(out["hit10_mean"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["l2_mac_k2", "l2_mac_k4", "l2_mac_k8"])
    a = ap.parse_args()
    T, B = Teacher(), Bench()
    dump("eg2_q4", B, B.top["synth_test"], B.top["real_test"], {"download_bytes": 179914969})
    for tag in a.tags:
        d = ROOT / "out" / "export" / tag
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        top = {t: B.ix.topk(ort_embed(d / "model_q4.onnx", tokenize(T, B.texts(t), remap=tok)))
               for t in ("synth_test", "real_test")}
        nbytes = json.loads((ROOT / "results" / f"export_{tag}.json").read_text())["formats"]["q4"]["bytes"]
        dump(tag, B, top["synth_test"], top["real_test"], {"download_bytes": nbytes, "file": f"out/export/{tag}/model_q4.onnx"})


if __name__ == "__main__":
    main()
