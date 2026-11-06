"""Tokenize TREC-COVID docs per arm for web/bridge_encode.js and put them on R2.

    python3 -I run/eval_tokens.py   # .work/evaltok/<arm>.trec.{ids,lens}.u16 -> R2 v1/evaltok/
"""
import sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from evalset import load  # noqa: E402
from r2 import put  # noqa: E402
W = HERE / ".work"
ARMS = {"stock": W / "e17c/epoch1/tokenizer.json", "ls32": W / "tok/ls32/tokenizer.json", "ls50": W / "tok/ls50/tokenizer.json"}
d, _, _ = load("trec-covid")
texts = [(a + " " + b).strip() for a, b in zip(d["title"], d["text"])]
(W / "evaltok").mkdir(exist_ok=True)
for arm, p in ARMS.items():
    t = Tokenizer.from_file(str(p)); t.no_padding(); t.enable_truncation(512)
    lens, ids = [], []
    for i in range(0, len(texts), 20000):
        enc = t.encode_batch(texts[i:i + 20000])
        lens += [len(e.ids) for e in enc]
        ids.append(np.concatenate([np.array(e.ids, np.uint16) for e in enc]))
    np.array(lens, np.uint16).tofile(W / f"evaltok/{arm}.trec.lens.u16")
    np.concatenate(ids).tofile(W / f"evaltok/{arm}.trec.ids.u16")
    for k in ("lens", "ids"):
        print(put(W / f"evaltok/{arm}.trec.{k}.u16", f"v1/evaltok/{arm}.trec.{k}.u16"))
    print(arm, len(lens), float(np.mean(lens)), flush=True)
