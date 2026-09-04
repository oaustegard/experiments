"""Mine (docstring -> code) training pairs from the AST chunks.

No issue text is used anywhere: the 59 test issues never touch training.
Anchor is the docstring; positive is the same chunk with the docstring cut
out, so the model has to bind prose to code rather than to itself. Held out
by file (5%) for a validation loss.
"""
from __future__ import annotations

import json
import random
import re

from common import HERE, load_chunks

DOC = re.compile(r'("""|\'\'\')(.*?)\1', re.S)


def mine() -> list[dict]:
    pairs = []
    for c in load_chunks("ast"):
        m = DOC.search(c["text"])
        if not m:
            continue
        doc = " ".join(m.group(2).split())
        if len(doc) < 40:
            continue
        code = (c["text"][: m.start()] + c["text"][m.end():]).strip()
        if len(code) < 60:
            continue
        pairs.append({"file": c["file"], "anchor": doc[:600], "positive": code[:2000]})
    return pairs


if __name__ == "__main__":
    random.seed(0)
    pairs = mine()
    files = sorted({p["file"] for p in pairs})
    random.shuffle(files)
    held = set(files[: max(1, len(files) // 20)])
    train = [p for p in pairs if p["file"] not in held]
    val = [p for p in pairs if p["file"] in held]
    (HERE / "data").mkdir(exist_ok=True)
    json.dump(train, open(HERE / "data" / "pairs_train.json", "w"))
    json.dump(val, open(HERE / "data" / "pairs_val.json", "w"))
    print(f"{len(pairs)} pairs over {len(files)} files -> train {len(train)} / val {len(val)}")
