"""Requantize the shipped decider's int8 matmul codes from +-63 to +-31 and score it.

    python3 -I run/codes31.py

Approximation: codes are halved and rounded from the shipped +-63 codes (q' = round(q / 2), scale x2),
not requantized from fp32, so the error is a little larger than a fresh +-31 build would have; a loss
here is an upper bound. Writes results/codes31.json with the score and the brotli size of the matmul codes.
"""
import json
import sys
from pathlib import Path

import brotli
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402
from weight_probe import MD, Enc, Probe, probs_of  # noqa: E402

man = json.loads((MD / "manifest.json").read_text())
data = bytearray((MD / "weights.bin").read_bytes())
old, new = [], []
for L in man["layers"]:
    for n in ("Wqkv", "Wo", "Wi", "Wo_mlp"):
        q, s = L[n]["q"], L[n]["s"]
        c = np.frombuffer(bytes(data[q["offset"]:q["offset"] + q["length"]]), np.int8)
        old.append(c.tobytes())
        c2 = np.clip(np.round(c / 2), -31, 31).astype(np.int8)
        new.append(c2.tobytes())
        data[q["offset"]:q["offset"] + q["length"]] = c2.tobytes()
        sc = np.frombuffer(bytes(data[s["offset"]:s["offset"] + s["length"]]), np.float32) * 2
        data[s["offset"]:s["offset"] + s["length"]] = sc.astype(np.float32).tobytes()
p = Probe(man, bytes(data))
tok = Enc(MD / "tokenizer.json")
te = examples([c for w in WORKFLOWS for c in load_cases(w, "test")])
for e in te:
    e["ids"], e["pos"] = tok.build(e)
summ, _ = score_examples(te, [x.tolist() for x in probs_of(p, te, man["temps"])])
out = {"test": summ, "shipped_accuracy": 0.73,
       "matmul_codes_brotli_63": len(brotli.compress(b"".join(old), quality=11)),
       "matmul_codes_brotli_31": len(brotli.compress(b"".join(new), quality=11)),
       "matmul_codes_raw": sum(map(len, old))}
(ROOT / "results" / "codes31.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
