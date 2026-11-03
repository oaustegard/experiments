"""Shared pieces for monad-weight-edit: model loading, the chat prompt, batched greedy
generation, answer parsing, loop detection.

Monad's chat format (model card, chat_template.json):
    <|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n
Every special-looking piece is ordinary text: `<|im_end|>` is five tokens, `</think>` three.
"""
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "monad"
END = "<|im_end|>"


def load(path=MODEL_DIR, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(str(path))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    model.eval()
    return tok, model


def prompt(q: str) -> str:
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


@torch.no_grad()
def generate(tok, model, prompts, max_new=1024, batch=16):
    """Greedy, batched, left-padded. Returns list of dicts with the generated ids (cut at the
    first <|im_end|> or EOS) and whether the generation stopped on its own."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                             stop_strings=[END], tokenizer=tok, cache_implementation="static")
        L = enc["input_ids"].shape[1]
        for row in gen[:, L:].tolist():
            out.append(cut(tok, row))
    return out


def cut(tok, row):
    """{ids, stopped}: generated ids up to the end marker. <|im_end|> is plain text tokenised in
    context, so it is found on the decoded string: the shortest prefix whose text contains it."""
    eos, pad = tok.eos_token_id, tok.pad_token_id
    if eos in row:
        return {"ids": row[:row.index(eos)], "stopped": True}
    while row and row[-1] == pad:
        row = row[:-1]
    if END not in tok.decode(row):
        return {"ids": row, "stopped": False}
    lo, hi = 1, len(row)
    while lo < hi:
        mid = (lo + hi) // 2
        if END in tok.decode(row[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return {"ids": row[:lo], "stopped": True}


def loop_period(ids, min_reps=3, max_period=64):
    """Smallest p such that the tail is the same p tokens repeated >= min_reps times, else None.
    (Same definition as weight-space-edit/lfm_eval.py.)"""
    n = len(ids)
    for p in range(1, max_period + 1):
        if p * min_reps > n:
            break
        tail = ids[n - p:]
        if all(ids[n - (k + 1) * p:n - k * p] == tail for k in range(1, min_reps)):
            return p
    return None


def split_think(text: str):
    """(trace, answer, closed). The prompt already opened <think>."""
    text = text.split(END, 1)[0]
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a, b.strip(), True
    return text, "", False


# ---------------------------------------------------------------- tasks
LETTERS = "ABCD"


def mmlu_question(x) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(x["choices"]))
    return f"{x['question'].strip()}\n\n{opts}"


def mmlu_parse(answer: str):
    """First standalone option letter in the post-think answer, else None."""
    m = re.search(r"\b([ABCD])\b[\.\):]?", answer)
    return m.group(1) if m else None


def gsm8k_gold(x) -> float:
    return float(x["answer"].split("####")[-1].strip().replace(",", ""))


def last_number(text: str):
    m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    s = m[-1].replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def loop_onset(ids, p):
    """Smallest s such that ids[s:] is p-periodic (ids[i] == ids[i + p] for every i >= s)."""
    n = len(ids)
    s = n - p
    while s > 0 and ids[s - 1] == ids[s - 1 + p]:
        s -= 1
    return s
