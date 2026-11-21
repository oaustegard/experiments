"""Pin a held-out SYNTH evaluation set for the teacher-forced metrics -> data/eval_synth.jsonl.

Rows come from shard synth_500.parquet, which no training run reads (training uses shards 001-0NN).
English only, stratified by exercise type because looping should help reasoning more than recall
(Inkling's loops left SimpleQA flat): 128 memorization, 128 math exercise, 128 mcq + math mcq.
Kept only if prompt + reasoning + answer fit in 1,900 tokens.

    python3 build_eval.py --parquet /path/synth_500.parquet
"""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SEED = 20261121
GROUPS = {"memorization": ["memorization"], "math": ["math exercise"], "mcq": ["mcq", "math mcq"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--per-group", type=int, default=128)
    ap.add_argument("--dev-per-group", type=int, default=32)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(HERE / "models/monad"))
    t = pq.read_table(a.parquet, columns=["synth_id", "language", "exercise", "query", "synthetic_reasoning",
                                          "synthetic_answer"]).to_pandas()
    t = t[(t.language == "en") & (t.synthetic_answer.str.strip() != "") & (t.synthetic_reasoning.str.strip() != "")]
    rng = random.Random(SEED)
    out = []
    for g, ex in GROUPS.items():
        sub = t[t.exercise.isin(ex)]
        idx = list(range(len(sub)))
        rng.shuffle(idx)
        n = 0
        for i in idx:
            r = sub.iloc[i]
            text = (f"<|im_start|>user\n{r.query}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    f"{r.synthetic_reasoning.strip()}\n</think>\n\n{r.synthetic_answer.strip()}<|im_end|>")
            if len(tok.encode(text, add_special_tokens=False)) > 1900:
                continue
            split = "dev" if n < a.dev_per_group else "test"
            out.append({"id": r.synth_id, "group": g, "exercise": r.exercise, "split": split, "q": r.query,
                        "reasoning": r.synthetic_reasoning.strip(), "answer": r.synthetic_answer.strip()})
            n += 1
            if n >= a.per_group + a.dev_per_group:
                break
    p = HERE / "data/eval_synth.jsonl"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({g: sum(1 for r in out if r["group"] == g) for g in GROUPS}, "rows ->", p)


if __name__ == "__main__":
    main()
