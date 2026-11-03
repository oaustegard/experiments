"""Generate on the pinned fixture with one model directory, checkpointing per batch.

    python3 run_eval.py --model models/monad --name base --split test

Writes runs/<name>__<split>.jsonl, one line per prompt; reruns skip uids already written.
Greedy, batch 32, max 1024 new tokens (1536 for IFEval, whose answers are essays).
"""
import argparse
import json
import time

import torch

from common import HERE, generate, load, loop_period, mmlu_question, prompt, split_think

MAX_NEW = {"mmlu": 1024, "gsm8k": 1024, "ifeval": 1536}


def question(item):
    r = item["row"]
    if item["task"] == "mmlu":
        return mmlu_question(r)
    if item["task"] == "gsm8k":
        return r["question"]
    return r["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tasks", default="mmlu,gsm8k,ifeval")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    fx = json.loads((HERE / "data/fixture.json").read_text())[a.split]
    fx = [x for x in fx if x["task"] in a.tasks.split(",")]
    if a.limit:
        fx = fx[:a.limit]
    out = HERE / "runs" / f"{a.name}__{a.split}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["uid"] for l in open(out)}
    todo = [x for x in fx if x["uid"] not in done]
    tok, model = load(HERE / a.model if not a.model.startswith("/") else a.model)
    print(f"{a.name}/{a.split}: {len(todo)} to do, {len(done)} done", flush=True)
    for task in ("mmlu", "gsm8k", "ifeval"):
        items = [x for x in todo if x["task"] == task]
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            t0 = time.time()
            res = generate(tok, model, [prompt(question(x)) for x in chunk], max_new=MAX_NEW[task], batch=a.batch)
            dt = time.time() - t0
            with open(out, "a") as f:
                for x, r in zip(chunk, res):
                    text = tok.decode(r["ids"])
                    _, _, closed = split_think(text)
                    f.write(json.dumps({"uid": x["uid"], "task": task, "n_tok": len(r["ids"]),
                                        "stopped": r["stopped"], "closed": closed,
                                        "loop": None if r["stopped"] else loop_period(r["ids"]),
                                        "text": text, "batch_s": round(dt, 1)}) + "\n")
            print(f"  {task} {i + len(chunk)}/{len(items)} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
