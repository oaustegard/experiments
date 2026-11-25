"""Ettin-17m [MASK] vectors for harvested candidates, on CPU, with browser-decider's NumPy reference.

Builds each sequence the way web/decider.js `build` does and takes the final hidden state at the
candidate's [MASK] position: the vector web/taste.js trains on. --check compares against vectors the
page computed on WebGPU (harvest lines that carry them).

    python3 run/embed.py results/harvest.jsonl results/vectors.npz [--check]
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP / "browser-decider" / "run"))
from pack import Ref  # noqa: E402

MODEL = EXP / "browser-decider" / "web" / "models" / "decider"
INSTR = "Which phrase should this radio station play next for this listener?"  # web/taste.js INSTR
MAX_LEN = 1024


class Encoder:
    def __init__(self):
        self.ref = Ref(json.loads((MODEL / "manifest.json").read_text()), (MODEL / "weights.bin").read_bytes())
        self.tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        self.CLS, self.SEP, self.MASK = (self.sp(t)[0] for t in ("[CLS]", "[SEP]", "[MASK]"))

    def sp(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def build(self, text, state):  # web/decider.js build("choice", INSTR, [["candidate", text]], state)
        ids = [self.CLS, *self.sp(f"choice: {INSTR}"), self.SEP]
        pos = len(ids)
        ids += [self.MASK, *self.sp(f" candidate: {text}"), self.SEP]
        budget = MAX_LEN - len(ids) - 1
        ids += [*self.sp(state)[:budget], self.SEP]
        return ids, pos

    def vec(self, text, state):
        ids, pos = self.build(text, state)
        return self.ref.hidden(np.array(ids))[pos].astype(np.float32)


def unb64(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    check = "--check" in sys.argv
    enc = Encoder()
    pts = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    desc, code, worst = [], [], 0.0
    for k, p in enumerate(pts):
        d = np.stack([enc.vec(c["text"], p["state"]) for c in p["info"]])
        c = np.stack([enc.vec(c["code"], p["state"]) for c in p["info"]])
        if check and p.get("desc"):
            for mine, theirs in ((d, p["desc"]), (c, p["code"])):
                ref = np.stack([unb64(s) for s in theirs])
                worst = max(worst, float(np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-9)))
        desc.append(d); code.append(c)
        if k % 25 == 0:
            print(f"{k + 1}/{len(pts)}", flush=True)
    np.savez_compressed(dst, desc=np.stack(desc), code=np.stack(code),
                        feats=np.array([[c["feats"] for c in p["info"]] for p in pts], dtype=np.float32))
    if check:
        print(f"max relative difference against the WebGPU vectors: {worst:.2e}")


if __name__ == "__main__":
    main()
