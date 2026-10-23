"""Token frequencies under the FULL tokenizer over training-side text only: posts, msmarco_train,
synth_train, gpu_more, and the caption-domain training texts (train-split alt + post text, harvested
alt text when present). Test queries are never counted. Writes .work/freq.npy (262,144 counts)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import LS, PREV, WORK, Teacher, tokenize  # noqa: E402

sys.path.insert(0, str(LS / "run"))
from img_common import IMG, train_texts  # noqa: E402


def main():
    T = Teacher()
    texts = []
    for l in (PREV / "out" / "posts.jsonl").open():
        texts.append(json.loads(l).get("text") or "")
    for n in ("msmarco_train", "synth_train", "gpu_more"):
        texts += [json.loads(l)["q"] for l in (PREV / "out" / "targets" / f"{n}.jsonl").open()]
    alts, posts = train_texts()
    texts += alts + posts
    for n in ("extra_alt", "extra_alt2"):
        p = IMG / f"{n}.json"
        if p.exists():
            texts += json.loads(p.read_text())
    c = np.zeros(262144, np.int64)
    for ids in tokenize(T, texts, prompt=""):
        np.add.at(c, ids, 1)
    np.save(WORK / "freq.npy", c)
    o = np.sort(c)[::-1]
    cum = np.cumsum(o) / o.sum()
    print(len(texts), "texts", int((c > 0).sum()), "distinct ids;", {k: round(float(cum[k - 1]), 4) for k in (8192, 32768, 41741, 65536, 131072)})


if __name__ == "__main__":
    main()
