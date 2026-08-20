#!/usr/bin/env python3
"""Route Gemini-authored NL for real commands, with a model under test.

This is the eval the whole thread was missing: the commands are real (cyber
corpus), the natural language is written by an independent model (`gen_nl.py`),
and neither side came from the model being scored. It measures the fine-tuned
gate model — and, with `--model`, the non-RAG ablation — on that clean eval.

Two numbers, both honest about their limits:

* **utility routing** — does the predicted command lead with the gold utility.
  Reported over all rows and over the leak-free subset (rows whose Gemini NL did
  not name the utility). The leak-free number is the one to quote.
* **functional equivalence** — for the subset that runs in `funceq.py`'s fixture,
  execute both and compare. Most cyber commands touch the network or security
  tools and cannot run here, so this is a small, explicitly-sized slice.

    python3 run_independent_eval.py --model ../nl2sh-retrieval/ft
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "nl2sh-retrieval"
PREFILL = ("<|language_start|>\nEnglish\n<|language_end|>\n"
           "<|query_report_start|>\nTrivial\n<|query_report_end|>\n<|answer_start|>\n")
CMDLINE = re.compile(r"`([^`\n]{2,200})`|^\s*([a-z][a-z0-9_.+-]{1,20}\s+[^\n]{2,200})$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nl", type=Path, default=HERE / "cyber_nl.json")
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--out", type=Path, default=HERE / "results_independent.json")
    a = ap.parse_args()

    sys.path.insert(0, str(GATE))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pleias_gate as G

    data = [r for r in json.loads(a.nl.read_text()) if r.get("nl")]
    tldr = G.load_tldr(a.tldr)
    data = [r for r in data if r["utility"] in tldr]  # gold must be answerable
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    import random
    rng = random.Random(20260819)
    others = [u for u in tldr if len(tldr[u]) >= 1]
    rows = []
    for r in data:
        gu = r["utility"]
        picks = [gu] + rng.sample([u for u in others if u != gu], a.distractors)
        rng.shuffle(picks)
        srcs = [f"{u} — {d}: {c}" for u in picks for d, c in tldr[u][:1]]
        ids = tok(G.build_prompt(r["nl"], srcs) + PREFILL, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        ans = re.split(r"<\|answer_end\|>", gen)[0].strip()
        m = CMDLINE.search(ans)
        cand = (m.group(1) or m.group(2) or "").strip() if m else ""
        rows.append({**r, "gold_cmd": r["cmd"], "command": cand,
                     "utility_ok": bool(cand) and G.gold_utility(cand) == gu,
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{'OK ' if rows[-1]['utility_ok'] else '   '} "
              f"{'[leak]' if r.get('names_utility') else '     '} "
              f"{r['nl'][:44]:<46} -> {cand[:44]!r}")

    clean = [r for r in rows if not r.get("names_utility")]
    n, nc = len(rows), len(clean)
    summary = {
        "model": a.model, "n": n, "n_leak_free": nc,
        "utility_acc_all": round(sum(r["utility_ok"] for r in rows) / n, 3),
        "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / nc, 3) if nc else None,
        "command_rate": round(sum(bool(r["command"]) for r in rows) / n, 3),
    }
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
