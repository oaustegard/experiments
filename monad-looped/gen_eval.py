"""Greedy generation on monad-weight-edit's pinned fixture for looped arms, in its run-file format.

    python3 gen_eval.py --configs configs.json --split dev [--model DIR] [--names a,b] [--limit N]

Writes runs/<config name>__<split>.jsonl (uid, task, n_tok, n_think, stopped, closed, loop, text);
reruns skip uids already written. Same settings as monad-weight-edit/run_eval.py: greedy, 1,024 new
tokens (1,536 for IFEval). Score with score.py, which wraps monad-weight-edit/summarize.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MWE = HERE.parent / "monad-weight-edit"
sys.path.insert(0, str(HERE))
from loopmodel import LoopedMonad, generate  # noqa: E402

MAX_NEW = {"mmlu": 1024, "gsm8k": 1024, "ifeval": 1536}
LETTERS = "ABCD"


def question(item):
    r = item["row"]
    if item["task"] == "mmlu":
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        return f"{r['question'].strip()}\n\n{opts}"
    return r["question"] if item["task"] == "gsm8k" else r["prompt"]


def prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def loop_period(ids, min_reps=3, max_period=64):
    n = len(ids)
    for p in range(1, max_period + 1):
        if p * min_reps > n:
            break
        tail = ids[n - p:]
        if all(ids[n - (k + 1) * p:n - k * p] == tail for k in range(1, min_reps)):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--model", default=str(HERE / "models/monad"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--names", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default="mmlu,gsm8k,ifeval")
    ap.add_argument("--max-new-scale", type=float, default=1.0)
    ap.add_argument("--runs", default=str(HERE / "runs"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"
    fx = json.loads((MWE / "data/fixture.json").read_text())[a.split]
    fx = [x for x in fx if x["task"] in a.tasks.split(",")]
    if a.limit:
        per = {}
        fx = [x for x in fx if per.setdefault(x["task"], []).append(1) or len(per[x["task"]]) <= a.limit]
    configs = json.loads(Path(a.configs).read_text())
    if a.names:
        configs = [c for c in configs if c["name"] in a.names.split(",")]
    kmax = max([c.get("K", 1) for c in configs] + [2])
    model = LoopedMonad.from_pretrained(a.model, band=None, kmax=kmax).to(dev).eval()
    learned_band = model.band
    for c in configs:
        model.band = tuple(c["band"]) if c.get("band") else None
        if model.band and "alpha" not in c:
            assert learned_band and tuple(learned_band) == model.band
        K, alpha = c.get("K", 1), c.get("alpha")
        out = Path(a.runs) / f"{c['name']}__{a.split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["uid"] for l in open(out)} if out.exists() else set()
        todo = [x for x in fx if x["uid"] not in done]
        print(f"{c['name']}/{a.split}: {len(todo)} to do, {len(done)} done", flush=True)
        for task in ("mmlu", "gsm8k", "ifeval"):
            items = [x for x in todo if x["task"] == task]
            for i in range(0, len(items), a.batch):
                chunk = items[i:i + a.batch]
                t0 = time.time()
                res = generate(model, tok, [prompt(question(x)) for x in chunk], K=K, alpha=alpha,
                               max_new=int(MAX_NEW[task] * a.max_new_scale), batch=a.batch, device=dev)
                dt = time.time() - t0
                with open(out, "a") as f:
                    for x, r in zip(chunk, res):
                        text = tok.decode(r["ids"])
                        closed = "</think>" in text
                        n_think = len(tok.encode(text.split("</think>")[0], add_special_tokens=False)) if closed else len(r["ids"])
                        f.write(json.dumps({"uid": x["uid"], "task": task, "n_tok": len(r["ids"]), "n_think": n_think,
                                            "stopped": r["stopped"], "closed": closed,
                                            "loop": None if r["stopped"] else loop_period(r["ids"]),
                                            "text": text, "batch_s": round(dt, 1)}) + "\n")
                print(f"  {c['name']} {task} {i + len(chunk)}/{len(items)} {dt:.0f}s", flush=True)
    print("done gen", flush=True)


if __name__ == "__main__":
    main()
