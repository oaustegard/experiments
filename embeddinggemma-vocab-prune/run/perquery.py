"""Per-query synth_test hit arrays for the board (experiments#104), plus the 32K cut's coverage.

Usage: python3 -I run/perquery.py

For each table-QAT arm (k=2) and table width, writes results/<arm>_perquery.json: 1,498 bools,
synth_test file order, true when the query's source post is in the top 10 of the committed EG2
q4 2-bit codes (ASYM). Transformer linears at q4 round-to-nearest (as shipped). Also records the
share of real_test / synth_test queries whose full-tokenizer ids all survive the 32K cut.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qat import LS, ROOT, VOCAB, quantized_copy  # noqa: E402
from bench import Bench  # noqa: E402
from model import Student, embed, tokenize  # noqa: E402
from common import Teacher  # noqa: E402

ARMS = {  # arm name: (checkpoint, table bits)
    "k2_base_tfp": (LS / "out/l1/k2_ep2_v32k/final.pt", 0),
    "k2_base_t3": (LS / "out/l1/k2_ep2_v32k/final.pt", 3),
    "k2_base_t2": (LS / "out/l1/k2_ep2_v32k/final.pt", 2),
    "k2_ctrl_tfp": (ROOT / ".work/qat/k2_ctrl/final.pt", 0),
    "k2_ctrl_t3": (ROOT / ".work/qat/k2_ctrl/final.pt", 3),
    "k2_ctrl_t2": (ROOT / ".work/qat/k2_ctrl/final.pt", 2),
    "k2_qat3_t3": (ROOT / ".work/qat/k2_b3/final.pt", 3),
    "k2_qat2_t2": (ROOT / ".work/qat/k2_b2/final.pt", 2),
}


def main():
    torch.set_num_threads(4)
    T = Teacher()
    rows = json.loads((VOCAB / "rows.json").read_text())
    tok = Tokenizer.from_file(str(VOCAB / "tokenizer.json"))
    B = Bench()
    enc = tokenize(T, B.texts("synth_test"), remap=tok)
    out = {}
    for arm, (ckpt, bits) in ARMS.items():
        m = Student(2, rows)
        assert not m.load_state_dict(torch.load(ckpt), strict=False).unexpected_keys
        q = quantized_copy(m, bits, True)
        top = B.ix.topk(embed(q, enc))
        hit = (top == B.src[:, None]).any(1)
        (ROOT / "results" / f"{arm}_perquery.json").write_text(json.dumps([bool(h) for h in hit]))
        out[arm] = float(hit.mean())
        print(arm, out[arm], flush=True)
    keep = set(rows)
    cov = {t: float(np.mean([all(i in keep for i in e) for e in tokenize(T, B.texts(t))]))
           for t in ["real_test", "synth_test"]}
    (ROOT / "results" / "perquery_summary.json").write_text(json.dumps(
        {"hit@10": out, "cut32k_fully_covered_share": cov, "metric": "ASYM synth_test hit@10, layers q4 RTN"}, indent=1))
    print(cov)


if __name__ == "__main__":
    main()
