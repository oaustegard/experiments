"""EG2 q4 text vectors (query prompt) for the image benchmark and the caption-domain training texts.

Usage: out/imgenv/bin/python -I run/img_teacher.py

Writes out/img/teacher_{eval_all,train_alt,train_post,extra_alt}.npy (extra_alt from run/img_harvest.py) with matching .json text lists.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_common import IMG, eval_queries, train_texts  # noqa: E402
from common import Teacher  # noqa: E402


def main():
    T = Teacher()
    ev, _ = eval_queries()
    alts, posts = train_texts()
    extra = json.loads((IMG / "extra_alt.json").read_text()) if (IMG / "extra_alt.json").exists() else []
    for name, texts in [("eval_all", ev), ("train_alt", alts), ("train_post", posts), ("extra_alt", extra)]:
        if not texts:
            continue
        p = IMG / f"teacher_{name}.npy"
        if p.exists():
            continue
        np.save(p, T.embed(texts))
        (IMG / f"teacher_{name}.json").write_text(json.dumps(texts, ensure_ascii=False))
        print(name, len(texts), flush=True)


if __name__ == "__main__":
    main()
