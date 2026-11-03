"""Premise check: does int4 weight quantization (onnxruntime-genai builder, MatMulNBits, block 32,
fp32 activations) keep Monad's predictions? Greedy 64 steps from 16 dev prompts with both graphs on
CPU onnxruntime; KL(fp32 || int4) per step while the two greedy paths agree, and how long they agree."""
import json, sys
import numpy as np
from common import HERE, load, prompt
from run_eval import question
import ort_gen

FX = json.loads((HERE / "data/fixture.json").read_text())["dev"]
tok, _ = load()
piece = json.load(open(HERE / "export/pieces.json"))
P = [tok.encode(prompt(question(x)), add_special_tokens=False) for x in FX[::12][:16]]
import os
REF = os.environ.get("REF", "base")
ref = ort_gen.generate(ort_gen.session(HERE / "export" / REF), P, 64, piece, record_logits=65, stop=False)
OUTP = HERE / os.environ.get("OUT", "results/q4_probe.json")
out = json.loads(OUTP.read_text()) if OUTP.exists() else {}
for name in sys.argv[1:]:
    r = ort_gen.generate(ort_gen.session(HERE / "export" / name), P, 64, piece, record_logits=65, stop=False)
    kls, agree = [], []
    for b in range(len(P)):
        n = 0
        while n < 64 and ref["tokens"][b][n] == r["tokens"][b][n]:
            n += 1
        agree.append(n)
        for s in range(n + 1):
            p = ref["logits"][s][b].astype(np.float64); q = r["logits"][s][b].astype(np.float64)
            p = p - p.max(); q = q - q.max()
            lp = p - np.log(np.exp(p).sum()); lq = q - np.log(np.exp(q).sum())
            kls.append(float((np.exp(lp) * (lp - lq)).sum()))
    out[name] = {"mean_kl": float(np.mean(kls)), "median_agree_steps": float(np.median(agree)),
                 "full_64_agree": int(sum(a == 64 for a in agree)), "n": len(P)}
    print(name, out[name], flush=True)
OUTP.write_text(json.dumps(out, indent=1))
