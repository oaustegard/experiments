"""Evaluate LFM2.5-350M, optionally with a weight edit, on GSM8K or chat prompts.

Each finished batch is appended to --out as JSON lines, so a killed run resumes where it stopped.

An edit file (torch.save of {"layers": {idx: [d1, d2, ...]}, ...}) removes each unit direction d from
the output of that layer's MLP down-projection: W2 <- (I - d d^T) W2. That is a change to the stored
weights, applied once at load, so the edited model is an ordinary checkpoint.
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM2.5-350M"
GSM_SUFFIX = "\nPlease reason step by step, and finish with the final number."
CHAT = [  # short spoken-assistant style requests, the voice-loop use
    "Tell me a fun fact about octopuses.", "What should I cook for dinner tonight?",
    "Explain how a bicycle gear works.", "Give me three tips for sleeping better.",
    "What's the difference between weather and climate?", "Write a short poem about autumn.",
    "How do I keep basil alive indoors?", "Why is the sky blue?",
    "Summarize the plot of Romeo and Juliet.", "What are some good stretches after a long bike ride?",
    "List the planets in order from the sun.", "How does compound interest work?",
    "Tell me a joke about computers.", "What is a black hole?",
    "Suggest a name for a grey cat.", "How do vaccines train the immune system?",
    "Count from one to twenty in words.", "Describe the water cycle.",
    "What is the capital of Australia, and why isn't it Sydney?", "Give me a recipe for pancakes.",
    "Translate 'good morning, how are you' into French, Spanish and German.",
    "What causes the seasons?", "Repeat after me: the quick brown fox jumps over the lazy dog.",
    "Name ten animals that live in the ocean.", "What's a good way to learn to play guitar?",
    "Explain what a neural network is to a ten-year-old.", "Why do cats purr?",
    "What are the primary colors?", "How long should I boil an egg?",
    "Tell me about the history of the bicycle.", "List the days of the week backwards.",
    "What is the Pythagorean theorem?",
]


def loop_period(ids, min_reps=3, max_period=64):
    """Smallest p such that the tail is the same p tokens repeated >= min_reps times, else None."""
    n = len(ids)
    for p in range(1, max_period + 1):
        if p * min_reps > n:
            break
        tail = ids[n - p:]
        if all(ids[n - (k + 1) * p:n - k * p] == tail for k in range(1, min_reps)):
            return p
    return None


def last_number(text):
    m = re.findall(r"boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    s = m[-1].replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def apply_edit(model, path):
    spec = torch.load(path)
    n = 0
    with torch.no_grad():
        for li, dirs in spec["layers"].items():
            W = model.model.layers[int(li)].feed_forward.w2.weight  # [hidden, inter]
            for d in dirs:
                d = (d / d.norm()).to(W.dtype)
                W -= torch.outer(d, d @ W)
                n += W.shape[1]
    return n


def load(edit=None):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    changed = apply_edit(model, edit) if edit else 0
    return tok, model, changed


def items(which, n):
    if which == "gsm8k":
        import pandas as pd
        from huggingface_hub import hf_hub_download
        df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
        for i, r in df.head(n).iterrows():
            gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
            yield {"id": int(i), "prompt": r["question"] + GSM_SUFFIX, "gold": gold}
    else:
        for i, p in enumerate(CHAT[:n]):
            yield {"id": i, "prompt": p, "gold": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["gsm8k", "chat"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--mode", choices=["greedy", "prod"], default="greedy")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--edit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [x for x in items(a.set, a.n) if x["id"] not in done]
    tok, model, changed = load(a.edit)
    print(f"edit={a.edit} weights changed={changed} todo={len(todo)}", flush=True)
    gen = dict(max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
    if a.mode == "greedy":
        gen.update(do_sample=False)
    else:  # Liquid's recommended settings, the ones voice-loop uses
        gen.update(do_sample=True, temperature=0.3, min_p=0.15, repetition_penalty=1.05)
    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) for x in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        torch.manual_seed(a.seed + chunk[0]["id"])
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **gen)
        dt = time.time() - t
        with open(a.out, "a") as f:
            for x, row in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
                eos = [i for i, v in enumerate(row) if v in (tok.eos_token_id, tok.pad_token_id)]
                ids = row[:eos[0]] if eos else row
                text = tok.decode(ids, skip_special_tokens=True)
                pred = last_number(text) if x["gold"] is not None else None
                rec = {"id": x["id"], "n_tok": len(ids), "hit_limit": not eos, "loop": loop_period(ids),
                       "pred": pred, "correct": None if x["gold"] is None else (pred is not None and abs(pred - x["gold"]) < 1e-6),
                       "text": text}
                f.write(json.dumps(rec) + "\n")
        print(f"batch {b // a.batch} {len(chunk)} items {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
