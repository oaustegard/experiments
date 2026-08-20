#!/usr/bin/env python3
"""Are the man/tldr pages actually in play? And what does the full pipeline score?

The 0.706 headline for Gemma was measured with ORACLE sources — the gold
utility's tldr example was guaranteed in context. That conflates three different
systems, which this script separates on the same independent cyber eval:

* **oracle** — gold utility's example always present (+2 distractors). The
  upper bound; what the bake-off reported.
* **none** — no sources at all. If this matches oracle, the model memorised the
  mapping and the retrieval tier is dead weight. If it drops, the pages are
  doing work. This is the diagnostic the question asks for.
* **retrieval** — the REAL pipeline: BM25 over the $PATH-scoped chunk corpus
  runs on the request, its top-k utilities' examples go to the model. This is
  the honest end-to-end number, and it is bounded above by oracle (retrieval
  sometimes misses the gold) — the cost of imperfect retrieval, priced.

The gap oracle→none measures how much the documentation contributes; the gap
oracle→retrieval measures how much our retrieval throws away.

    python3 gemma_fullsystem.py --model ft_gemma --tldr <tldr>/pages
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELFHIST = HERE.parent / "nl2sh-selfhist"
sys.path.insert(0, str(HERE))
import pleias_gate as G
import retrieve as R
from gemma_arm import build_user


def retrieved_sources(index: R.Index, tldr: dict, nl: str, k: int) -> list[str]:
    """Top-k distinct utilities from a real BM25 query, one example each."""
    seen, srcs = [], []
    for chunk, _ in index.search(nl, k=15):
        if chunk.utility in seen or chunk.utility not in tldr:
            continue
        seen.append(chunk.utility)
        d, c = tldr[chunk.utility][0]
        srcs.append(f"{chunk.utility} — {d}: {c}")
        if len(seen) >= k:
            break
    return srcs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=HERE / "ft_gemma")
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, default=HERE / "data" / "chunks.jsonl")
    ap.add_argument("--nl", type=Path, default=SELFHIST / "cyber_nl.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", type=Path, default=SELFHIST / "results_fullsystem.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tldr = G.load_tldr(a.tldr)
    data = [r for r in json.loads(a.nl.read_text()) if r.get("nl") and r["utility"] in tldr]
    index = R.Index(R.load_chunks(a.chunks))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    rng = random.Random(a.seed)

    def run(mode: str) -> dict:
        rows = []
        for r in data:
            gu = r["utility"]
            if mode == "oracle":
                others = [u for u in tldr if len(tldr[u]) >= 1 and u != gu]
                picks = [gu] + rng.sample(others, a.distractors)
                rng.shuffle(picks)
                srcs = [f"{u} — {tldr[u][0][0]}: {tldr[u][0][1]}" for u in picks]
            elif mode == "none":
                srcs = []
            else:  # retrieval
                srcs = retrieved_sources(index, tldr, r["nl"], a.k)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": build_user(r["nl"], srcs)}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            cmd = gen.strip().strip("`").split("\n")[0].strip()
            if cmd.startswith("bash"):
                cmd = cmd[4:].strip()
            gold_in_src = any(s.split(" ")[0] == gu for s in srcs)
            rows.append({"utility": gu, "nl": r["nl"], "command": cmd,
                         "names_utility": r.get("names_utility", False),
                         "utility_ok": bool(cmd) and G.gold_utility(cmd) == gu,
                         "gold_in_sources": gold_in_src})
        clean = [r for r in rows if not r["names_utility"]]
        return {"mode": mode, "n": len(rows), "n_leak_free": len(clean),
                "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / len(clean), 3),
                "gold_in_sources_rate": round(sum(r["gold_in_sources"] for r in rows) / len(rows), 3),
                "rows": rows}

    out = {m: run(m) for m in ("oracle", "none", "retrieval")}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n{'mode':<12}{'routing (leak-free)':>22}{'gold in sources':>18}")
    print("-" * 52)
    for m in ("oracle", "none", "retrieval"):
        s = out[m]
        print(f"{m:<12}{s['utility_acc_leak_free']:>22.3f}{s['gold_in_sources_rate']:>18.3f}")
    print(f"\ncontribution of docs (oracle - none): "
          f"{out['oracle']['utility_acc_leak_free'] - out['none']['utility_acc_leak_free']:+.3f}")
    print(f"cost of real retrieval (oracle - retrieval): "
          f"{out['oracle']['utility_acc_leak_free'] - out['retrieval']['utility_acc_leak_free']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
