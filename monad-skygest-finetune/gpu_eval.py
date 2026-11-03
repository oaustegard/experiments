"""Greedy answers to every prompt in data/eval_prompts.jsonl for each checkpoint, batched on GPU
(HF transformers, fp32). Used to choose a checkpoint; the reported numbers come from the 8-bit export
on onnxruntime (evalgen.py), which is what the page runs.

    python3 gpu_eval.py runs_gpu models/A8p ckpt/lr1e-4-e2 ckpt/lr1e-4-e4 ...
    SETS=papers,facts python3 gpu_eval.py runs_sel ckpt/...     # only those probe sets
Writes <outdir>/<checkpoint name>.jsonl. Skips checkpoints already written.
"""
import json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
END = "<|im_end|>"


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


@torch.no_grad()
def run(path, items, out, batch=48):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(dev).eval()
    items = sorted(items, key=lambda x: (x["max_new"], len(x["q"])))
    t0 = time.time()
    with open(out, "w") as f:
        for i in range(0, len(items), batch):
            ch = items[i:i + batch]
            enc = tok([prompt(x["q"]) for x in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**enc, max_new_tokens=max(x["max_new"] for x in ch), do_sample=False,
                               eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                               stop_strings=[END], tokenizer=tok)
            L = enc["input_ids"].shape[1]
            for x, row in zip(ch, g[:, L:].tolist()):
                row = row[:x["max_new"]]
                stopped = False
                if tok.eos_token_id in row:
                    row, stopped = row[:row.index(tok.eos_token_id)], True
                while row and row[-1] == tok.pad_token_id:
                    row = row[:-1]
                text = tok.decode(row)
                if END in text:
                    text, stopped = text.split(END, 1)[0], True
                closed = "</think>" in text
                trace, ans = (text.split("</think>", 1) + [""])[:2] if closed else (text, "")
                r = {k: x[k] for k in ("set", "id") + (("task",) if "task" in x else ())}
                r.update({"stopped": stopped, "closed": closed, "tokens": len(row),
                          "loop": (not stopped) and loop_period(row) is not None, "answer": ans.strip(), "trace": trace})
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {Path(path).name} {i + len(ch)}/{len(items)} {time.time() - t0:.0f}s", flush=True)


def main():
    outdir = HERE / sys.argv[1]
    outdir.mkdir(parents=True, exist_ok=True)
    import os
    sets = set(os.environ.get("SETS", "papers,facts,dev").split(","))
    items = [x for x in map(json.loads, open(HERE / "data/eval_prompts.jsonl")) if x["set"] in sets]
    for path in sys.argv[2:]:
        out = outdir / f"{Path(path).name}.jsonl"
        if out.exists():
            continue
        run(path, items, out.with_suffix(".part"))
        out.with_suffix(".part").rename(out)   # a file under its final name is complete


if __name__ == "__main__":
    main()
