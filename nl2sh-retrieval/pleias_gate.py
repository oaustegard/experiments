#!/usr/bin/env python3
"""Does a 350M RAG model produce a usable shell command from retrieved documentation?

This is the gate. `monad-bsky` ran the equivalent check on Pleias Monad (56M) and got
**0 parseable calls out of 62** — the model analysed the instruction instead of
answering it — which killed a fine-tuning plan before it was paid for. Same check,
one generation on: Pleias-RAG-350M is trained for grounded answers quoted from
supplied sources, which is precisely the operation Monad failed (it copied an
identifier correctly 51% of the time against a purpose-built model's 78-90%).

The test deliberately supplies a *perfect* retriever: the gold utility's tldr
examples are always among the sources. That isolates the model. If it cannot pick
the right command when the right command is sitting in its context, no retrieval
tier will save it.

Four things are measured, in increasing order of what they would let us build:

1. **parse rate** — does an `<|answer_start|>...<|answer_end|>` span appear at all?
   This is the number that was 0.000 for Monad.
2. **command rate** — does that span contain something shaped like a shell command?
3. **utility accuracy** — is it the right utility?
4. **verbatim rate** — is the command copied character-for-character from a source,
   rather than regenerated? This is the span-copying property the whole architecture
   rests on, because a model that retypes `--max-depth` as `--max-dept` is unusable
   no matter how well it routes.

The model's I/O protocol is its special tokens, not a chat template: the prompt is
`<|query_start|>...<|query_end|>` followed by `<|source_start|><|source_id|>N
...<|source_end|>` blocks, and generation emits a fixed reasoning scaffold
(language, query analysis, source analysis, draft) before the answer span. That
structure is why the formatting failure mode does not apply here — the NL2SH paper
found markdown parsing alone worth +21 to +25 points on sub-7B models, and this
model delimits its answer explicitly instead.

    python3 pleias_gate.py --n 3 --smoke      # verify format, measure latency
    python3 pleias_gate.py --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "PleIAs/Pleias-RAG-350M"

# Pre-filling the reasoning scaffold and starting generation at the answer span is
# a 9x latency win: 61.2 s for the full trace against 5.4 s here, on 4 CPU cores,
# because the language/query-analysis/source-analysis/draft preamble is ~700 tokens
# and none of it is the answer. It does not change what the model says.
PREFILL = ("<|language_start|>\nEnglish\n<|language_end|>\n"
           "<|query_report_start|>\nTrivial\n<|query_report_end|>\n<|answer_start|>\n")

ANSWER = re.compile(r"<\|answer_start\|>(.*?)(?:<\|answer_end\|>|$)", re.S)
# a command-ish line: a bare token followed by args, or anything inside backticks
CMDLINE = re.compile(r"`([^`\n]{2,200})`|^\s*([a-z][a-z0-9_.+-]{1,20}\s+[^\n]{2,200})$", re.M)


def parse_tldr(page: Path) -> list[tuple[str, str]]:
    """(description, command) pairs from one tldr markdown page."""
    out, desc = [], None
    for line in page.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("- ") and s.endswith(":"):
            desc = s[2:-1]
        elif s.startswith("`") and s.endswith("`") and desc:
            cmd = s.strip("`")
            # tldr marks slots as {{placeholder}}; keep the literal text, drop the braces
            out.append((desc, re.sub(r"\{\{(.*?)\}\}", r"\1", cmd)))
            desc = None
    return out


def load_tldr(pages_dir: Path) -> dict[str, list[tuple[str, str]]]:
    by_util: dict[str, list[tuple[str, str]]] = {}
    for p in pages_dir.rglob("*.md"):
        ex = parse_tldr(p)
        if ex:
            by_util.setdefault(p.stem, []).extend(ex)
    return by_util


def gold_utility(cmd: str) -> str:
    m = re.sub(r'"[^"]*"|\'[^\']*\'', '""', cmd)
    seg = re.split(r"\|\||&&|\||;", m)[0]
    for tok in seg.split():
        if tok in ("sudo", "time", "nohup", "command", "!"):
            continue
        if "=" in tok and not tok.startswith("-"):
            continue
        return tok.strip("()`$")
    return ""


def build_prompt(query: str, sources: list[str]) -> str:
    """Exactly the format `Pleias-RAG-Library/RAGWithCitations._format_prompt` emits.

    Two details are load-bearing and both were wrong on the first attempt, which
    produced a 0.0 parse rate that looked like a capability verdict and was not:
    every block ends with a newline, and the prompt must END with
    `<|language_start|>\n`. Without that trailing token the model has no signal
    that the source list is closed, so it keeps emitting `<|source_start|>` blocks
    and degenerates into repetition. Reading the reference implementation cost two
    minutes; inferring it from the special-token list cost a false negative.
    """
    prompt = f"<|query_start|>{query}<|query_end|>\n"
    for i, s in enumerate(sources, 1):
        prompt += f"<|source_start|><|source_id|>{i} {s}<|source_end|>\n"
    return prompt + "<|language_start|>\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tldr", type=Path, required=True, help="tldr pages/ directory")
    ap.add_argument("--nl2bash", type=Path, required=True, help="nl2bash data/bash directory")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--examples-per-util", type=int, default=1)
    ap.add_argument("--no-prefill", action="store_true",
                    help="generate the full reasoning scaffold instead of skipping to the answer")
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--smoke", action="store_true", help="print full generations")
    ap.add_argument("--model-path", default=None,
                    help="local fine-tuned checkpoint; defaults to the HF base model")
    ap.add_argument("--out", type=Path, default=HERE / "results_gate.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    nls = (a.nl2bash / "all.nl").read_text(encoding="utf-8", errors="replace").splitlines()
    cms = (a.nl2bash / "all.cm").read_text(encoding="utf-8", errors="replace").splitlines()
    tldr = load_tldr(a.tldr)
    print(f"tldr: {len(tldr)} utilities with parsed examples")

    rng = random.Random(a.seed)
    pool = [(n, c) for n, c in zip(nls, cms) if gold_utility(c) in tldr]
    rng.shuffle(pool)
    rows = pool[: a.n]
    print(f"eval rows: {len(rows)} (gold utility always has tldr examples)\n")

    src = a.model_path or MODEL
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()

    others = [u for u in tldr if len(tldr[u]) >= 2]
    recs, lat = [], []
    for i, (nl, cm) in enumerate(rows, 1):
        gu = gold_utility(cm)
        picks = [gu] + rng.sample([u for u in others if u != gu], a.distractors)
        rng.shuffle(picks)
        sources, src_cmds = [], []
        for u in picks:
            for d, c in tldr[u][:a.examples_per_util]:
                sources.append(f"{u} — {d}: {c}")
                src_cmds.append(c)

        prompt = build_prompt(nl, sources) + ("" if a.no_prefill else PREFILL)
        ids = tok(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        dt = time.perf_counter() - t0
        lat.append(dt)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)

        if a.no_prefill:
            am = ANSWER.search(gen)
            ans = am.group(1).strip() if am else ""
            parsed = bool(am)
        else:
            ans = re.split(r"<\|answer_end\|>", gen)[0].strip()
            parsed = bool(ans)
        cand = ""
        if ans:
            cm_m = CMDLINE.search(ans)
            if cm_m:
                cand = (cm_m.group(1) or cm_m.group(2) or "").strip()
        recs.append({
            "nl": nl, "gold_cmd": cm, "gold_utility": gu,
            "n_sources": len(sources), "prompt_tokens": int(ids["input_ids"].shape[1]),
            "parsed": parsed, "answer": ans[:400], "command": cand,
            "utility_ok": bool(cand) and gold_utility(cand) == gu,
            "verbatim": bool(cand) and any(cand in s for s in src_cmds),
            "seconds": round(dt, 1),
        })
        if a.smoke:
            print(f"--- {i} ---\nQUERY: {nl[:110]}\nGOLD:  {cm[:110]}\n"
                  f"PROMPT TOKENS: {ids['input_ids'].shape[1]}  GEN {dt:.1f}s\n"
                  f"RAW: {gen[:700]}\n")
        else:
            print(f"{i:>3}/{len(rows)} {dt:>5.1f}s  parsed={recs[-1]['parsed']!s:<5} "
                  f"util_ok={recs[-1]['utility_ok']!s:<5} verbatim={recs[-1]['verbatim']!s:<5} "
                  f"{cand[:60]}")

    n = len(recs)
    summary = {
        "model": src, "n": n, "distractors": a.distractors,
        "examples_per_util": a.examples_per_util, "prefill": not a.no_prefill,
        "n_sources": recs[0]["n_sources"],
        "names_gold_utility": round(sum(
            re.search(rf"\b{re.escape(r['gold_utility'])}\b", r["answer"]) is not None
            for r in recs) / n, 3),
        "parse_rate": round(sum(r["parsed"] for r in recs) / n, 3),
        "command_rate": round(sum(bool(r["command"]) for r in recs) / n, 3),
        "utility_acc": round(sum(r["utility_ok"] for r in recs) / n, 3),
        "verbatim_rate": round(sum(r["verbatim"] for r in recs) / n, 3),
        "median_seconds": round(sorted(lat)[len(lat) // 2], 1),
        "median_prompt_tokens": sorted(r["prompt_tokens"] for r in recs)[n // 2],
    }
    a.out.write_text(json.dumps({"summary": summary, "rows": recs}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
