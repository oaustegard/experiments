"""Compare a bridge `check` result (logits of every step) with ort_gen.py on CPU.

    python3 check_parity.py bridge/jobs/<id>/out.json      # local SwiftShader run
    python3 check_parity.py --fetched DIR                   # gpu_bridge.py fetch output dir
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

import ort_gen
from common import load, prompt

HERE = Path(__file__).resolve().parent


def main():
    if sys.argv[1] == "--fetched":
        d = Path(sys.argv[2])
        res = json.loads((d / "result.json").read_text())
        r = res.get("outputs", res)
        logits = np.load(d / "logits.npy") if (d / "logits.npy").exists() else None
    else:
        out = json.loads(Path(sys.argv[1]).read_text())
        r = out["result"]
        logits = np.frombuffer(base64.b64decode(r["logits"]["b64"]), dtype=np.float32)
    S, B, V = r["shape"]
    logits = logits.reshape(S, B, V)
    tok, _ = load()
    qs = json.loads((HERE / "bridge/check_prompts.json").read_text())
    P = [tok.encode(prompt(q), add_special_tokens=False) for q in qs]
    sess = ort_gen.session(HERE / "export/base")
    ref = ort_gen.generate(sess, P, S - 1, ort_gen.pieces(tok), record_logits=S, stop=False)
    refl = np.stack(ref["logits"])
    d = np.abs(refl - logits)
    agree = (refl.argmax(-1) == logits.argmax(-1)).mean()
    same_tokens = sum(a == b for a, b in zip(ref["tokens"], r["tokens"]))
    print(json.dumps({"steps": S, "batch": B, "max_abs_logit_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                      "argmax_agree": float(agree), "identical_sequences": f"{same_tokens}/{B}",
                      "gpu_step_ms": r.get("stepMs", [])[:5], "load": r.get("load")}, indent=1))


if __name__ == "__main__":
    main()
