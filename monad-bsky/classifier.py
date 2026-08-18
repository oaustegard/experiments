#!/usr/bin/env python3
"""Score Monad over the 18 declared names instead of letting it generate one.

Oskar's third proposal, and the one that addresses two measured deficits at
once. Free generation lets the fine-tuned model emit `get_posts`,
`search_followers`, once `get_spammer.bsky.social` — names that were never
declared, on 14.5% of queries — and gives no score to gate on.

Scoring inverts that. For each query, prefill the prompt once, then evaluate
`log P(tokens of "<name>" | prompt)` for each of the 18 declared names and take
the argmax. Two things follow by construction:

* an undeclared name is unreachable, exactly as a decode grammar would make it,
  without needing a grammar;
* the softmax over the 18 candidates is a **confidence signal** — a margin
  between the top two — which the generative arm does not have at all.

    python3 classifier.py --model tuned-e2

Arguments still have to come from somewhere: this scores the *name* only, so
argument values are taken from the query by the same regex fill as `repair.py`.
Length-normalised and raw scores are both reported, because names differ in
token count and argmax over a raw sum quietly prefers short names.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

from monad_bsky.prompt import build_prompt

NEEDLE = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE))


def load_items() -> list[dict]:
    return [
        json.loads(x) for x in (NEEDLE / "evalset.jsonl").read_text().splitlines() if x.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tuned-e2")
    ap.add_argument("--label", default=None)
    ap.add_argument("--arm", default="tuned-min")
    ap.add_argument(
        "--empty-trace",
        action="store_true",
        help="score the name with no reasoning generated first (off-distribution; see ERRORS.md)",
    )
    a = ap.parse_args()

    import torch
    from needle_bsky.router import load_schemas
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from repair import repair_args

    label = a.label or f"classifier-{a.model}"
    schemas = load_schemas(a.arm)
    names = [s["name"] for s in schemas]
    items = load_items()

    model_dir = str(HERE / a.model)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    # The completion the trained model produces is `<reasoning>\n</think>\n{"name": "X", ...}`.
    # Score each candidate at the position where the name is emitted, with the
    # reasoning skipped: the prefix below is what the model sees just before it
    # commits to a name.
    PREFIX = '</think>\n{"name": "'
    cand_ids = [tok(n, add_special_tokens=False)["input_ids"] for n in names]

    rows, lat = [], []
    for n_done, it in enumerate(items, 1):
        base = build_prompt(schemas, it["query"])
        t0 = time.perf_counter()
        if a.empty_trace:
            prompt = base + PREFIX
        else:
            # The model was trained to derive first and name second, so scoring
            # the name against an empty <think> block asks it for something it
            # never saw. Let it write its own trace, then score the name at the
            # position it would actually have emitted one.
            gen_ids = tok(base, return_tensors="pt")
            with torch.no_grad():
                g = model.generate(
                    **gen_ids,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            trace = tok.decode(g[0][gen_ids["input_ids"].shape[1]:], skip_special_tokens=False)
            trace = trace.split("</think>", 1)[0]
            prompt = base + trace + PREFIX
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            last = out.logits[:, -1]
            scores = []
            for cid in cand_ids:
                lp = torch.log_softmax(last, dim=-1)[0, cid[0]].item()
                if len(cid) > 1:
                    step = model(
                        input_ids=torch.tensor([cid[:-1]]),
                        past_key_values=past,
                        use_cache=False,
                    )
                    lps = torch.log_softmax(step.logits[0], dim=-1)
                    lp += sum(lps[k, cid[k + 1]].item() for k in range(len(cid) - 1))
                scores.append(lp)
        lat.append((time.perf_counter() - t0) * 1000)

        norm_scores = [s / len(c) for s, c in zip(scores, cand_ids)]
        order = sorted(range(len(names)), key=lambda k: -scores[k])
        order_n = sorted(range(len(names)), key=lambda k: -norm_scores[k])
        top, second = order[0], order[1]
        mx = max(scores)
        probs = [pow(2.718281828, s - mx) for s in scores]
        z = sum(probs)
        conf = probs[top] / z

        chosen = names[top]
        args = repair_args(it["query"], {k: None for k in _required(schemas, chosen)})
        args = {k: v for k, v in args.items() if v is not None}
        accepted = it["tool"]
        tool_ok = chosen in accepted if accepted else False
        args_ok = tool_ok and all(
            _norm(args.get(k)) == _norm(v) for k, v in it.get("args", {}).items()
        )
        rows.append(
            {
                "id": it["id"],
                "cat": it["cat"],
                "query": it["query"],
                "expected": accepted,
                "got": chosen,
                "got_length_normalised": names[order_n[0]],
                "arguments": args,
                "tool_ok": tool_ok,
                "tool_ok_length_normalised": names[order_n[0]] in accepted if accepted else False,
                "args_ok": args_ok,
                "in_top3": any(names[k] in accepted for k in order[:3]) if accepted else False,
                "confidence": round(conf, 4),
                "margin": round(scores[top] - scores[second], 4),
                "latency_ms": round(lat[-1], 1),
            }
        )
        if n_done % 20 == 0:
            print(f"  {n_done}/{len(items)}", flush=True)

    on = [r for r in rows if r["expected"]]
    off = [r for r in rows if not r["expected"]]
    summary = {
        "n": len(rows),
        "tool_acc_routable": round(sum(r["tool_ok"] for r in on) / len(on), 4),
        "tool_acc_routable_length_normalised": round(
            sum(r["tool_ok_length_normalised"] for r in on) / len(on), 4
        ),
        "top3_routable": round(sum(r["in_top3"] for r in on) / len(on), 4),
        "args_acc_routable": round(sum(r["args_ok"] for r in on) / len(on), 4),
        "hallucinated_tool_rate": 0.0,
        "refusal_acc": 0.0,
        "note": "scoring over declared names cannot refuse; every query gets a call",
        "n_off_topic_forced_to_call": len(off),
        "median_latency_ms": round(statistics.median(lat), 1),
    }
    res = {"label": label, "model": model_dir, "summary": summary, "rows": rows}
    (HERE / f"results_{label}.json").write_text(json.dumps(res, indent=1))

    s = summary
    print(
        f"{label}: routable {s['tool_acc_routable']:.3f}  "
        f"(length-normalised {s['tool_acc_routable_length_normalised']:.3f})  "
        f"top-3 {s['top3_routable']:.3f}  args {s['args_acc_routable']:.3f}  "
        f"median {s['median_latency_ms']:.0f}ms"
    )
    # gate sweep on the softmax confidence
    print("  confidence gate over the 18-way softmax:")
    for t in (0.0, 0.3, 0.5, 0.7, 0.9):
        sel = [r for r in on if r["confidence"] >= t]
        if sel:
            print(
                f"    conf >= {t:.1f}  coverage {len(sel) / len(on):.3f}  "
                f"precision {sum(r['tool_ok'] for r in sel) / len(sel):.3f}"
            )
    return 0


def _required(schemas, name):
    for s in schemas:
        if s["name"] == name:
            return s["parameters"].get("required", [])
    return []


def _norm(v):
    return " ".join(str(v).strip().lstrip("@").lower().split())


if __name__ == "__main__":
    raise SystemExit(main())
