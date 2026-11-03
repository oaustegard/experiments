"""Pin the evaluation rows (not just a seed) into data/fixture.json.

test: MMLU 200 (seeded sample of the 14,042 test rows), GSM8K test rows 0-149, IFEval 150 (seeded
sample of 541). dev: disjoint rows of the same three sets, used only to design the edit.
"""
import json
import random

from common import HERE, read_jsonl

SEED = 20261103


def main():
    mm = read_jsonl(HERE / "data/mmlu_test.jsonl")
    gs = read_jsonl(HERE / "data/gsm8k_test.jsonl")
    ie = read_jsonl(HERE / "data/ifeval.jsonl")
    rng = random.Random(SEED)
    mm_idx = rng.sample(range(len(mm)), 264)
    ie_idx = rng.sample(range(len(ie)), 214)
    fx = {"seed": SEED, "test": [], "dev": []}

    def add(split, task, i, row):
        fx[split].append({"uid": f"{task}:{i}", "task": task, "row": row})

    for i in mm_idx[:200]:
        add("test", "mmlu", i, mm[i])
    for i in range(150):
        add("test", "gsm8k", i, gs[i])
    for i in ie_idx[:150]:
        add("test", "ifeval", i, ie[i])
    for i in mm_idx[200:]:
        add("dev", "mmlu", i, mm[i])
    for i in range(1000, 1064):
        add("dev", "gsm8k", i, gs[i])
    for i in ie_idx[150:]:
        add("dev", "ifeval", i, ie[i])
    (HERE / "data/fixture.json").write_text(json.dumps(fx))
    print({k: len(v) for k, v in fx.items() if k != "seed"})


if __name__ == "__main__":
    main()
