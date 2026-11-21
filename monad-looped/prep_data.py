"""Tokenize SYNTH shards into the flat arrays train.py reads.

    python3 prep_data.py --shards 1-4 --out data/train_tok [--cache DIR]

English rows with a non-empty reasoning and answer, from synth_001.parquet ... (never shard 500, the eval
shard). Each row: prompt + `reasoning\\n</think>\\n\\nanswer<|im_end|>` + EOS; target_start marks where the
loss begins (after the prompt). Rows over 2,048 tokens are dropped. Writes ids.npy (int16), offsets.npy,
target_start.npy and stats.json.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
URL = "https://huggingface.co/datasets/PleIAs/SYNTH/resolve/main/synth_{:03d}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1-4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/synth")
    ap.add_argument("--model", default=str(HERE / "models/monad"))
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    lo, hi = (int(x) for x in a.shards.split("-"))
    assert not (lo <= 500 <= hi), "shard 500 is the eval shard"
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    all_ids, offsets, tstart = [], [0], []
    n_rows = n_drop = 0
    eos = tok.eos_token_id
    for s in range(lo, hi + 1):
        p = Path(a.cache) / f"synth_{s:03d}.parquet"
        if not p.exists():
            subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(p), URL.format(s)], check=True)
        t = pq.read_table(p, columns=["language", "query", "synthetic_reasoning", "synthetic_answer"]).to_pandas()
        t = t[(t.language == "en") & (t.synthetic_answer.str.strip() != "") & (t.synthetic_reasoning.str.strip() != "")]
        for c in range(0, len(t), 5000):   # chunked: a whole shard of Python token lists does not fit in 15 GB
            tc = t.iloc[c:c + 5000]
            prompts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n" for q in tc["query"]]
            targets = [f"{r.strip()}\n</think>\n\n{an.strip()}<|im_end|>" for r, an in zip(tc.synthetic_reasoning, tc.synthetic_answer)]
            P = tok(prompts, add_special_tokens=False)["input_ids"]
            T = tok(targets, add_special_tokens=False)["input_ids"]
            for p_ids, t_ids in zip(P, T):
                n = len(p_ids) + len(t_ids) + 1
                if n > a.max_len:
                    n_drop += 1
                    continue
                all_ids.append(np.asarray(p_ids + t_ids + [eos], dtype=np.int16))
                offsets.append(offsets[-1] + n)
                tstart.append(len(p_ids))
                n_rows += 1
            del P, T, prompts, targets
        del t
        print(f"shard {s}: rows {n_rows} dropped {n_drop} tokens {offsets[-1]:,}", flush=True)
        p.unlink()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ids.npy", np.concatenate(all_ids))
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "target_start.npy", np.asarray(tstart, dtype=np.int32))
    stats = {"shards": a.shards, "rows": n_rows, "dropped": n_drop, "tokens": int(offsets[-1])}
    (out / "stats.json").write_text(json.dumps(stats))
    print("done prep", stats, flush=True)


if __name__ == "__main__":
    main()
