"""Round 4 real-question pairs: MedQuAD (NIH consumer-health questions -> their answers, CC BY 4.0).

    python3 -I run/pairs_medquad.py   # .work/shards/mq.{ids,lens}.u16, mq_pairs.npz; results/pairs_mq_stats.json

Source: lavita/MedQuAD on the Hub (47,441 rows; the answers of three sources were removed upstream for
copyright, which leaves 16,374 with an answer). Query = e17c's query prompt + the question; positive = the
answer's first passage (sentences packed to <= 90 words, as run/data.py), at most 128 tokens with Ettin's
tokenizer. Rows are written q, d, q, d, ...; the pair order is a seeded shuffle per epoch, cut into steps
of 64 pairs, enough epochs to cover --steps.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from data import passages  # noqa: E402
from model import QP  # noqa: E402

W = HERE / ".work"
S = W / "shards"
P, STEPS = 64, 1000


def main():
    f = hf_hub_download("lavita/MedQuAD", "data/train-00000-of-00001-e36383d177026d53.parquet", repo_type="dataset")
    d = pd.read_parquet(f)
    d = d[d.answer.fillna("").str.len() > 50]
    d = d.drop_duplicates(["question", "answer"])
    tok = Tokenizer.from_file(str(W / "e17c/epoch1/tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(128)
    texts = []
    for q, a in zip(d.question, d.answer):
        ps = passages(" ".join(a.split()))
        texts += [QP + " ".join(q.split()), ps[0]]
    enc = tok.encode_batch(texts)
    lens = np.array([len(e.ids) for e in enc], np.uint16)
    ids = np.concatenate([np.array(e.ids, np.uint16) for e in enc])
    ids.tofile(S / "mq.ids.u16")
    lens.tofile(S / "mq.lens.u16")
    n = len(texts) // 2
    rng = np.random.default_rng(1)
    order = []
    while len(order) * P < STEPS * P:
        perm = rng.permutation(n)
        for i in range(0, n - P + 1, P):
            b = perm[i:i + P]
            order.append(np.stack([2 * b, 2 * b + 1], 1))
    order = np.concatenate(order[:STEPS]).astype(np.uint32)
    np.savez(S / "mq_pairs.npz", order=order)
    stats = {"source": "lavita/MedQuAD", "pairs": n, "steps": STEPS, "epochs": round(STEPS * P / n, 2),
             "mean_len_q": float(lens[0::2].mean()), "mean_len_d": float(lens[1::2].mean()),
             "truncated_frac_d": float((lens[1::2] == 128).mean()), "by_source": d.document_source.value_counts().to_dict()}
    (HERE / "results/pairs_mq_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
