"""Sweep LoopCD guidance strength (and loop depth) on Inkling 2 Mini.

Greedy decoding, the same problems under every setting, so each setting is
compared to the unguided baseline problem by problem. Every (problem, setting)
result is appended to a JSONL file as it finishes; rerun the same command to
resume after a stop.

    python sweep.py --n 100                       # GSM8K, omega 0/0.1/0.25/0.5
    python sweep.py --settings 0,0.25,p2:0,p2:0.25 --n 100
    python sweep.py --data my.jsonl --effort medium --max-tokens 16384

A setting is `omega` (3 passes) or `pN:omega` (N loop passes).
`--data` takes JSONL rows with "question" and a numeric "answer".
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
INSTRUCTION = (
    "\n\nSolve the problem. On the last line write the final answer as a "
    "number after '####'."
)


def load_problems(data: str | None, n: int) -> list[dict]:
    if data is None:
        path = HERE / "data" / "gsm8k_test.jsonl"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(GSM8K_URL, path)
    else:
        path = Path(data)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = []
    for i, r in enumerate(rows[:n]):
        gold = str(r["answer"]).split("####")[-1]
        out.append({"id": r.get("id", i), "question": r["question"], "gold": _num(gold)})
    return out


def _num(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract(text: str) -> float | None:
    final = text.split("<|content_text|>")[-1]
    for pattern in (rf"####\s*\$?({NUMBER})", rf"\\boxed\{{\s*({NUMBER})"):
        hits = re.findall(pattern, final)
        if hits:
            return _num(hits[-1])
    hits = re.findall(NUMBER, final)
    return _num(hits[-1]) if hits else None


def parse_setting(s: str) -> tuple[int | None, float]:
    if s.startswith("p"):
        passes, omega = s[1:].split(":")
        return int(passes), float(omega)
    return None, float(s)


def mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar (binomial test on the discordant pairs)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(records: list[dict], settings: list[str]) -> dict:
    by = {s: {r["id"]: r for r in records if r["setting"] == s} for s in settings}
    base = by.get("0", {})
    summary = {}
    for s in settings:
        rows = list(by[s].values())
        if not rows:
            continue
        entry = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "truncated": sum(r["finish"] == "length" for r in rows),
            "mean_gen_tokens": statistics.mean(r["gen_tokens"] for r in rows),
            "median_tok_s": statistics.median(r["tok_s"] for r in rows),
        }
        if s != "0" and base:
            shared = [i for i in by[s] if i in base]
            wins = sum(by[s][i]["correct"] and not base[i]["correct"] for i in shared)
            losses = sum(base[i]["correct"] and not by[s][i]["correct"] for i in shared)
            entry |= {
                "paired_n": len(shared),
                "wins_vs_0": wins,
                "losses_vs_0": losses,
                "mcnemar_p": round(mcnemar_p(wins, losses), 4),
            }
        summary[s] = entry
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="mlx-community/Inkling-2-Mini-4bit")
    p.add_argument("--settings", default="0,0.1,0.25,0.5")
    p.add_argument("--data", default=None, help="JSONL with question/answer; default GSM8K test")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--effort", default="low", help="Inkling reasoning_effort")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--hidden-only", action="store_true", help="don't guide block states")
    p.add_argument("--out", default=str(HERE / "results" / "sweep.jsonl"))
    a = p.parse_args()

    import loopcd
    from mlx_lm import load, stream_generate

    settings = a.settings.split(",")
    problems = load_problems(a.data, a.n)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    done = {(r["id"], r["setting"]) for r in records}

    model, tok = load(a.model)
    for k, prob in enumerate(problems):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prob["question"] + INSTRUCTION}],
            add_generation_prompt=True,
            reasoning_effort=a.effort,
        )
        for s in settings:
            if (prob["id"], s) in done:
                continue
            passes, omega = parse_setting(s)
            loopcd.enable(model, omega, passes=passes, guide_blocks=not a.hidden_only)
            text, last, t0 = [], None, time.time()
            for r in stream_generate(model, tok, prompt, max_tokens=a.max_tokens):
                text.append(r.text)
                last = r
            text = "".join(text)
            pred = extract(text)
            rec = {
                "id": prob["id"],
                "setting": s,
                "gold": prob["gold"],
                "pred": pred,
                "correct": pred is not None and prob["gold"] is not None
                and abs(pred - prob["gold"]) < 1e-6,
                "gen_tokens": last.generation_tokens,
                "tok_s": round(last.generation_tps, 2),
                "prompt_tok_s": round(last.prompt_tps, 1),
                "peak_gb": round(last.peak_memory, 2),
                "finish": last.finish_reason,
                "seconds": round(time.time() - t0, 1),
                "tail": text[-300:],
            }
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            records.append(rec)
            print(
                f"[{k + 1}/{len(problems)}] {s:>8} {'ok ' if rec['correct'] else 'BAD'} "
                f"{rec['gen_tokens']:>5} tok {rec['tok_s']:>6} tok/s",
                flush=True,
            )
        loopcd.disable(model)

    summary = summarize(records, settings)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
