"""Score a run file: quality per task and the failure modes.

    python3 summarize.py runs/base__test.jsonl [more ...] [--json results/x.json]

Failure modes, per prompt:
  limit   the generation hit max_new without <|im_end|> or EOS
  loop    hit the limit AND the tail is one block of 1-64 tokens repeated >= 3 times
  open    the think block never closed (no </think>)
Quality:
  mmlu    first standalone A-D after </think>; no answer counts as wrong
  gsm8k   the number after the last "answer" in the post-think text, else its first number
  ifeval  google's checker (lm_eval's port) on the post-think text, prompt-level strict and loose
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import gsm8k_gold, mmlu_parse, split_think  # noqa: E402

FX = None


def fixture():
    global FX
    if FX is None:
        f = json.loads((HERE / "data/fixture.json").read_text())
        FX = {x["uid"]: x for s in ("test", "dev") for x in f[s]}
    return FX


NUM = r"-?\$?\s?\d[\d,]*(?:\.\d+)?"


def gsm8k_answer(ans: str):
    """The number after the last 'answer' keyword, else the first number of the answer text."""
    low = ans.lower()
    k = low.rfind("answer")
    for seg in ([ans[k:]] if k >= 0 else []) + [ans]:
        m = re.search(NUM, seg)
        if m:
            s = m.group(0).replace("$", "").replace(",", "").replace(" ", "")
            try:
                return float(s)
            except ValueError:
                pass
    return None


def score(rec):
    x = fixture()[rec["uid"]]
    trace, ans, closed = split_think(rec["text"])
    t = rec["task"]
    out = {"limit": not rec["stopped"], "loop": rec.get("loop") is not None, "open": not closed,
           "n_tok": rec["n_tok"]}
    if t == "mmlu":
        got = mmlu_parse(ans) if closed else None
        out["correct"] = got == "ABCD"[x["row"]["answer"]]
        out["parsed"] = got is not None
    elif t == "gsm8k":
        got = gsm8k_answer(ans) if closed else None
        out["correct"] = got is not None and abs(got - gsm8k_gold(x["row"])) < 1e-6
        out["parsed"] = got is not None
    else:
        import lm_eval.tasks.ifeval.utils as u
        r = u.process_results(x["row"] | {"key": x["row"]["key"]}, [ans if closed else ""])
        out["correct"] = bool(r["prompt_level_strict_acc"])
        out["loose"] = bool(r["prompt_level_loose_acc"])
        out["inst_strict"] = r["inst_level_strict_acc"]
        out["parsed"] = closed
    return out


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    by = {}
    for r in recs:
        by.setdefault(r["task"], []).append(score(r))
    res = {}
    for t, xs in by.items():
        n = len(xs)
        d = {"n": n, "acc": sum(x["correct"] for x in xs) / n,
             "limit": sum(x["limit"] for x in xs) / n, "loop": sum(x["loop"] for x in xs) / n,
             "open": sum(x["open"] for x in xs) / n, "parsed": sum(x["parsed"] for x in xs) / n,
             "mean_tok": sum(x["n_tok"] for x in xs) / n}
        if t == "ifeval":
            d["loose"] = sum(x["loose"] for x in xs) / n
            inst = [v for x in xs for v in x["inst_strict"]]
            d["inst_strict"] = sum(inst) / len(inst)
        res[t] = {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
    allx = [x for xs in by.values() for x in xs]
    res["all"] = {"n": len(allx), "limit": round(sum(x["limit"] for x in allx) / len(allx), 4),
                  "loop": round(sum(x["loop"] for x in allx) / len(allx), 4),
                  "open": round(sum(x["open"] for x in allx) / len(allx), 4),
                  "tokens": sum(x["n_tok"] for x in allx)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {Path(p).stem: summarize(p) for p in a.runs}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
