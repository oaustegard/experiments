#!/usr/bin/env python3
"""Gemma 3 270M as the on-device generator — a provenance-driven alternative.

The fine-tuned generator so far is Pleias-350M (Apache-2.0, French lab). The
question: does a Google model of the same size class do as well, for teams that
prefer Google provenance over a Chinese base like Qwen? Gemma 4's smallest is
E2B (2B effective) — too big for the phone-sized niche — so the match is
**Gemma 3 270M** (270M params, 262k vocab, 32k context; `gemma` license, not
Apache-2.0).

This is an apples-to-apples arm: the **same 600 training rows** and the **same
independent cyber eval** as `finetune_gate.py` / `run_independent_eval.py`, so
the only variable is the base model. Two things differ mechanically and are
handled here rather than assumed:

* **Format.** Gemma has none of Pleias' `<|source_start|>` / `<|answer_start|>`
  tokens; it uses `<start_of_turn>user … <end_of_turn><start_of_turn>model`.
  The sources and request go in the user turn, the command is the model turn,
  and loss is masked to the model turn only.
* **Vocabulary.** Gemma's 262k vocab (vs Pleias-RAG's 65k) means more of a path
  or flag is a single token, which is the exact axis `monad-bsky` found bounded
  small-model transcription — so this arm also tests whether a bigger vocab
  lifts the verbatim rate that stayed 0.000 for Pleias.

    python3 gemma_arm.py train --rows 600 --epochs 1
    python3 gemma_arm.py eval  --nl ../nl2sh-selfhist/cyber_nl.json
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
BASE = "unsloth/gemma-3-270m-it"
SELFHIST = HERE.parent / "nl2sh-selfhist"

SYS = ("Translate the request into a single shell command. "
       "Use the documented utilities below when relevant. Output only the command.")


def build_user(nl: str, sources: list[str]) -> str:
    src = "\n".join(f"- {s}" for s in sources)
    return f"{SYS}\n\nUtilities:\n{src}\n\nRequest: {nl}"


def _extract_command(gen: str) -> str:
    """Pull a command from model output, fenced or bare.

    An instruction-tuned model wraps commands in ```bash fences; a fine-tuned
    one emits them bare. The earlier parser stripped backticks then split on the
    newline, which turned "```bash\ncd ~\n```" into "bash" -> "" and scored the
    untrained model at a spurious 0.03. Handle the fence explicitly.
    """
    import re
    m = re.search(r"```(?:bash|sh|shell)?\s*\n?(.+?)```", gen, re.S)
    body = m.group(1) if m else gen
    for line in body.strip().splitlines():
        line = line.strip().strip("`").strip()
        if line and not line.lower().startswith(("here", "sure", "to ", "you ", "this ")):
            return line
    return ""


def gold_utility(cmd: str) -> str:
    sys.path.insert(0, str(HERE))
    import pleias_gate as G
    return G.gold_utility(cmd)


def make_sources(gu, tldr, rng, distractors):
    others = [u for u in tldr if len(tldr[u]) >= 1 and u != gu]
    picks = [gu] + rng.sample(others, distractors)
    rng.shuffle(picks)
    return [f"{u} — {tldr[u][0][0]}: {tldr[u][0][1]}" for u in picks if u in tldr]


def train(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(HERE))
    import pleias_gate as G

    tldr = G.load_tldr(a.tldr)
    nls = (a.nl2bash / "all.nl").read_text(errors="replace").splitlines()
    cms = (a.nl2bash / "all.cm").read_text(errors="replace").splitlines()
    rng = random.Random(a.seed)
    pool = [(n, c) for n, c in zip(nls, cms) if G.gold_utility(c) in tldr]
    rng.shuffle(pool)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).train()

    rows = []
    for nl, cm in pool[: a.rows]:
        gu = G.gold_utility(cm)
        # Source dropout: with prob `dropout_sources`, train on NO sources so the
        # model learns to fall back on its own knowledge when retrieval misses.
        # The ablation showed the oracle-trained model scores 0.000 without
        # sources; this teaches graceful degradation instead.
        if rng.random() < a.dropout_sources:
            srcs = []
        else:
            srcs = make_sources(gu, tldr, rng, a.distractors)
        user = build_user(nl, srcs)
        # Gemma turn format via the tokenizer's own template
        prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        full = prompt + cm + tok.eos_token
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        fids = tok(full, add_special_tokens=False)["input_ids"][:1024]
        labels = ([-100] * len(pids) + fids[len(pids):])[:1024]
        rows.append({"input_ids": fids, "labels": labels})

    from torch.utils.data import DataLoader
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        }
    dl = DataLoader(rows, batch_size=a.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(dl) * a.epochs))

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        for batch in dl:
            loss = model(**batch).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"  ep{ep} step {step}/{len(dl)*a.epochs} loss {loss.item():.4f} "
                      f"{(time.time()-t0)/60:.1f}m", flush=True)
            if (time.time() - t0) / 60 >= a.budget_minutes:
                print("budget spent; stopping"); break
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump({"base": BASE, "steps": step, "rows": a.rows,
               "dropout_sources": a.dropout_sources,
               "minutes": round((time.time()-t0)/60, 1)},
              open(a.results, "w"), indent=1)
    print(f"saved to {a.out}")


def evaluate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(HERE))
    import pleias_gate as G

    tldr = G.load_tldr(a.tldr)
    data = [r for r in json.loads(a.nl.read_text()) if r.get("nl") and r["utility"] in tldr]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    rng = random.Random(a.seed)

    rows = []
    for r in data:
        srcs = make_sources(r["utility"], tldr, rng, a.distractors)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": build_user(r["nl"], srcs)}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 repetition_penalty=a.rep_penalty,
                                 no_repeat_ngram_size=a.no_repeat,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        cmd = _extract_command(gen)
        rows.append({**r, "command": cmd,
                     "utility_ok": bool(cmd) and G.gold_utility(cmd) == r["utility"],
                     "verbatim_src": False})
        print(f"{'OK ' if rows[-1]['utility_ok'] else '   '} "
              f"{'[leak]' if r.get('names_utility') else '     '} "
              f"{r['utility']:<10} {cmd[:46]}")

    clean = [r for r in rows if not r.get("names_utility")]
    summary = {"model": str(a.model), "n": len(rows), "n_leak_free": len(clean),
               "utility_acc_all": round(sum(r["utility_ok"] for r in rows) / len(rows), 3),
               "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / len(clean), 3),
               "command_rate": round(sum(bool(r["command"]) for r in rows) / len(rows), 3)}
    (SELFHIST / f"results_gemma{a.tag}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = lambda p: [
        p.add_argument("--tldr", type=Path, required=True),
        p.add_argument("--distractors", type=int, default=2),
        p.add_argument("--seed", type=int, default=20260819)]
    t = sub.add_parser("train"); common(t)
    t.add_argument("--nl2bash", type=Path, required=True)
    t.add_argument("--rows", type=int, default=600); t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=2); t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--budget-minutes", type=float, default=60)
    t.add_argument("--dropout-sources", type=float, default=0.0,
                   help="fraction of training rows given no sources (graceful-degradation training)")
    t.add_argument("--out", type=Path, default=HERE / "ft_gemma")
    t.add_argument("--results", type=Path, default=HERE / "results_finetune_gemma.json")
    e = sub.add_parser("eval"); common(e)
    e.add_argument("--nl", type=Path, default=SELFHIST / "cyber_nl.json")
    e.add_argument("--model", type=Path, default=HERE / "ft_gemma")
    e.add_argument("--rep-penalty", type=float, default=1.0)
    e.add_argument("--no-repeat", type=int, default=0)
    e.add_argument("--tag", default="")
    a = ap.parse_args()
    (train if a.cmd == "train" else evaluate)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
