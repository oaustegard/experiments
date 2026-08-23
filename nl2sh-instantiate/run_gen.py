#!/usr/bin/env python3
"""Run one model under one prompt condition over the independent cyber eval.

Everything except the prompt is held at stage 1's settings: the same
gold-plus-two-distractors source construction, the same seed, the same greedy
decode with no repetition penalty, the same fenced-or-bare command parser. The
sources are **oracle** — the gold utility is always among them — because §6 of
`nl2sh-dense/RESULTS.md` isolates prompt effects that way, and because issue #52
puts retrieval out of scope: at 0.555 gold-in-sources against an 0.640 oracle
ceiling, the loss being measured here is the generator's.

    python3 run_gen.py --condition instantiate --model unsloth/gemma-3-270m-it \
        --tldr /path/to/tldr/pages --out results_it_instantiate.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RETRIEVAL = HERE.parent / "nl2sh-retrieval"
SELFHIST = HERE.parent / "nl2sh-selfhist"
DENSE = HERE.parent / "nl2sh-dense"
sys.path.insert(0, str(RETRIEVAL))
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402

DEFAULT_EVALS = [SELFHIST / "cyber_nl.json", DENSE / "cyber_nl_ext.json"]


def extract_command(gen: str) -> str:
    """gemma_arm._extract_command, unchanged — a fenced or a bare command."""
    m = re.search(r"```(?:bash|sh|shell)?\s*\n?(.+?)```", gen, re.S)
    body = m.group(1) if m else gen
    for line in body.strip().splitlines():
        line = line.strip().strip("`").strip()
        if line and not line.lower().startswith(("here", "sure", "to ", "you ", "this ")):
            return line
    return ""


def load_eval(paths: list[Path], tldr: dict) -> list[dict]:
    rows, seen = [], set()
    for p in paths:
        for r in json.loads(p.read_text()):
            key = (r.get("nl"), r.get("cmd"))
            if not r.get("nl") or r["utility"] not in tldr or key in seen:
                continue
            seen.add(key)
            rows.append({**r, "source_file": p.name})
    return rows


def make_sources(gu: str, tldr: dict, rng: random.Random, distractors: int) -> list[str]:
    """gemma_arm.make_sources — gold plus n distractors, shuffled, first example each."""
    others = [u for u in tldr if len(tldr[u]) >= 1 and u != gu]
    picks = [gu] + rng.sample(others, distractors)
    rng.shuffle(picks)
    return [f"{u} — {tldr[u][0][0]}: {tldr[u][0][1]}" for u in picks if u in tldr]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="weight dtype. stage 1 ran float32; a 4B model does not fit 15 GB at float32.")
    ap.add_argument("--condition", required=True, choices=sorted(prompts.BUILDERS))
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--nl", type=Path, nargs="*", default=DEFAULT_EVALS)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pleias_gate as G

    tldr = G.load_tldr(a.tldr)
    data = load_eval(list(a.nl), tldr)[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
    build = prompts.BUILDERS[a.condition]
    rng = random.Random(a.seed)

    rows, t_start = [], time.perf_counter()
    for r in data:
        srcs = make_sources(r["utility"], tldr, rng, a.distractors)
        user = build(r["nl"], srcs)
        text = tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 repetition_penalty=a.rep_penalty,
                                 no_repeat_ngram_size=a.no_repeat,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        dt = time.perf_counter() - t0
        n_prompt = ids["input_ids"].shape[1]
        n_new = int(out.shape[1] - n_prompt)
        gen = tok.decode(out[0][n_prompt:], skip_special_tokens=True)
        cmd = extract_command(gen)
        rows.append({**r, "gold_cmd": r["cmd"], "command": cmd, "raw": gen.strip()[:400],
                     "utility_ok": bool(cmd) and G.gold_utility(cmd) == r["utility"],
                     "sources": srcs, "literals": prompts.literals(r["nl"]),
                     "prompt_tokens": n_prompt, "new_tokens": n_new,
                     "seconds": round(dt, 2)})
        print(f"{'OK ' if rows[-1]['utility_ok'] else '   '}"
              f"{'[leak]' if r.get('names_utility') else '      '} "
              f"{r['utility']:<12} {cmd[:56]}", flush=True)

    clean = [r for r in rows if not r.get("names_utility")]
    tot_new = sum(r["new_tokens"] for r in rows)
    summary = {
        "model": str(a.model), "condition": a.condition, "dtype": a.dtype, "sources": "oracle",
        "distractors": a.distractors, "seed": a.seed, "n": len(rows),
        "n_leak_free": len(clean),
        "utility_acc_all": round(sum(r["utility_ok"] for r in rows) / len(rows), 3),
        "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / len(clean), 3),
        "command_rate": round(sum(bool(r["command"]) for r in rows) / len(rows), 3),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
        "mean_new_tokens": round(tot_new / len(rows), 1),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "decode_tok_per_s": round(tot_new / sum(r["seconds"] for r in rows), 1),
        "wall_minutes": round((time.perf_counter() - t_start) / 60, 1),
    }
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
