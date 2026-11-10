"""Label states with Jev: one call per row, every named question in that call.

    python3 run/jev_label.py IN.jsonl OUT.jsonl [--workers 8] [--limit N]

IN rows: {"id", "state" (string or object), "questions": [names from suite.ALL_QS]}.
OUT rows: {"id", "probs": {question: {option: p}}, "tokens"} or {"id", "error"}. Resumable: ids already
in OUT with probs are skipped. Prints one progress line per 500 rows and `done -> OUT (n ok, m failed)`.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite import ALL_QS, jev_probs, jev_question

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"


def call(row):
    qs = {n: jev_question(ALL_QS[n]) for n in row["questions"]}
    body = json.dumps({"model": MODEL, "state": row["state"], "questions": qs}).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"], "Content-Type": "application/json"})
    err = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                res = json.load(r)
            probs = {n: jev_probs(ALL_QS[n], res["answers"][n]) for n in row["questions"]}
            return {"id": row["id"], "probs": probs, "tokens": (res.get("usage") or {}).get("input_tokens")}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:160]!r}"
            if e.code in (400, 413, 422):
                break
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError) as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"
        time.sleep(1.5 * 2 ** attempt)
    return {"id": row["id"], "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with open(a.inp) as f:
        rows = [json.loads(line) for line in f]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {r["id"] for r in map(json.loads, out.read_text().splitlines()) if "probs" in r}
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    ok = bad = toks = 0
    t0 = time.time()
    with out.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(call, todo), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            ok += "probs" in r
            bad += "error" in r
            toks += r.get("tokens") or 0
            if i % 500 == 0:
                print(f"{i}/{len(todo)} ok {ok} failed {bad} tokens {toks} {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"done -> {out} ({ok + len(done)} ok, {bad} failed, {toks} input tokens this run)", flush=True)


if __name__ == "__main__":
    main()
