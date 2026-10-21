"""Per-query arrays for the 10-epoch Mac builds at each table width, in eg2-layer-student's format.

Usage: python3 -I run/perquery_mac.py

For k in 2, 4, 8 (release eg2-layer-mac-20261020, unpacked under .work/mac) and table bits 4, 3, 2
(asymmetric groups of 32, after training), with transformer linears at q4 round-to-nearest,
writes results/mac_k<k>_t<bits>_perquery.json = {"synth_hit10": 1,498 bools in synth_test order,
"real_overlap10": 1,000 floats in real_test order}: the same keys as
eg2-layer-student/results/l2_mac_k*_perquery.json, so the two pair query by query (ASYM).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qat import ROOT, VOCAB, quantized_copy  # noqa: E402
from bench import TESTS, Bench  # noqa: E402
from model import Student, embed, tokenize  # noqa: E402
from common import Teacher  # noqa: E402


def main():
    torch.set_num_threads(4)
    T = Teacher()
    rows = json.loads((VOCAB / "rows.json").read_text())
    tok = Tokenizer.from_file(str(VOCAB / "tokenizer.json"))
    B = Bench()
    enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    summary = {}
    for k in (2, 4, 8):
        m = Student(k, rows)
        sd = torch.load(ROOT / ".work" / "mac" / "out" / "l1" / f"mac_k{k}_v32k" / "final.pt")
        assert not m.load_state_dict(sd, strict=False).unexpected_keys
        for bits in (4, 3, 2):
            q = quantized_copy(m, bits, True)
            _, per = B.score({t: embed(q, enc[t]) for t in TESTS})
            top = B.ix.topk(embed(q, enc["synth_test"]))
            hit = (top == B.src[:, None]).any(1)
            out = {"synth_hit10": [bool(h) for h in hit], "real_overlap10": [float(x) for x in per["real_test"]]}
            (ROOT / "results" / f"mac_k{k}_t{bits}_perquery.json").write_text(json.dumps(out))
            summary[f"mac_k{k}_t{bits}"] = {"hit@10": float(hit.mean()), "real_overlap@10": float(np.mean(per["real_test"]))}
            print(k, bits, json.dumps(summary[f"mac_k{k}_t{bits}"]), flush=True)
    (ROOT / "results" / "perquery_mac_summary.json").write_text(json.dumps(
        {"metric": "ASYM, layers q4 RTN, table asymmetric groups of 32 after training", **summary}, indent=1))


if __name__ == "__main__":
    main()
