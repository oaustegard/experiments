#!/usr/bin/env python3
"""Ask Jev, per lint hit, P(the hit is a real tic worth rewriting).

    python3 jev_judge.py A 1     # arm A, run 1 -> data/jev_A_1.jsonl
    python3 jev_judge.py B 2

Arm A: category, lint note, flagged text, paragraph.
Arm B: A plus the register.md entry for the category.
Resumable: items already in the output file are skipped.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"
QUESTION = ("Is the flagged text, in this paragraph, a real instance of the writing tic described: "
            "one a careful human technical writer editing this paragraph would rewrite? Answer no when "
            "the regex matched but the construction does real work here: a triad of three real items, a "
            "clause that carries a fact, a certainty word the evidence earns, a quoted or technical use.")


def state(item, arm):
    s = {"task": "A regex lint for machine-sounding prose flagged this passage in a published technical "
                 "blog post. Decide whether the flag is right.",
         "tic_category": item["category"], "lint_note": item["note"],
         "flagged_text": item["match_text"], "paragraph": item["paragraph"]}
    if arm == "B":
        s["tic_definition"] = item["entry"]
    return s


def call(item, arm):
    body = {"model": MODEL, "state": state(item, arm),
            "questions": {"tic": {"type": "noul", "instructions": QUESTION}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"], "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.load(r)
            return {"id": item["id"], "p": float(res["answers"]["tic"]["noul"]),
                    "ms": round((time.monotonic() - t0) * 1000), "usage": res.get("usage")}
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 ** attempt)
    return {"id": item["id"], "p": None, "error": err}


def main():
    arm, run = sys.argv[1], sys.argv[2]
    items = [json.loads(l) for l in (HERE / "data/items.jsonl").read_text().splitlines()]
    out = HERE / f"data/jev_{arm}_{run}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [i for i in items if i["id"] not in done]
    with out.open("a") as fh, ThreadPoolExecutor(8) as ex:
        for r in ex.map(lambda i: call(i, arm), todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    print(f"{out.name}: {len(rows)} rows, {sum(r['p'] is None for r in rows)} errors")


if __name__ == "__main__":
    main()
