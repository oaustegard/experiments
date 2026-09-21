"""Builds data/train_ids.npy: N sequences x 256 token ids from the wikitext-103 raw TRAIN split,
packed the same way as data.py (consecutive non-heading lines, truncated to 256, [CLS]..[SEP]),
in corpus order. Regenerable, so gitignored; data/train_meta.json is tracked.
Usage: python3 data_train.py [--n 24000]. Completion line: 'done -> data/train_ids.npy'."""
import argparse, json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SEQ = 256
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24000); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
out = []; buf = []
for line in ds["text"]:
    s = line.strip()
    if not s or s.startswith("="):
        continue
    buf.append(s)
    ids = tok(" ".join(buf), add_special_tokens=True)["input_ids"]
    if len(ids) >= SEQ:
        ids = ids[:SEQ]; ids[-1] = tok.sep_token_id
        out.append(ids); buf = []
        if len(out) >= a.n:
            break
arr = np.array(out, dtype=np.int32)
assert (arr[:, 0] == tok.cls_token_id).all() and (arr[:, -1] == tok.sep_token_id).all()
(HERE / "data").mkdir(exist_ok=True)
np.save(HERE / "data" / "train_ids.npy", arr)
json.dump({"n": int(arr.shape[0]), "seq_len": SEQ, "tokens": int(arr.size), "source": "Salesforce/wikitext wikitext-103-raw-v1 train, in order",
           "tokenizer": "answerdotai/ModernBERT-base"}, open(HERE / "data" / "train_meta.json", "w"), indent=1)
print(f"sequences {arr.shape[0]} tokens {arr.size}")
print("done -> data/train_ids.npy")
