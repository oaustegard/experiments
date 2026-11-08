#!/usr/bin/env python3
"""Score every attempt with Jev, one binary question per call.

Jev is called at api.typesafe.ai with a pinned model id (the same backend as the Muninn `decide`
tool; scripts cannot reach the MCP tool). Inputs are the issue text, the diff, the worker summary and
the visible-test output. The held-out result, the arm, the replicate and the run path never reach the
judge: diff headers are rewritten to task-relative paths.

Dev scoring is free. Test scoring refuses to run unless `prompts_frozen.json` carries the sha256 of the
current QUESTIONS, so the prompts cannot change after the test split has been read.

  JEV_PROMPT_VERSION=v2 python3 score_jev.py --split dev  --out data/jev/v2
  python3 score_jev.py --freeze                      # records the prompt hash
  python3 score_jev.py --split test --out data/jev/v1
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "jev-1.14.0"
URL = "https://api.typesafe.ai/v1/systemone"

QUESTIONS_V1 = {
    "Q1": "A worker fixed a bug in a small Python repository. `issue` is the bug report it was given, "
          "`diff` is its final change, `worker_summary` is what it reported, and `visible_test_output` is "
          "the result of the visible tests it could run. The code is also graded by a larger hidden test "
          "suite you cannot see. Answer yes if the task is fully solved: the change repairs the underlying "
          "cause everywhere it occurs, so the hidden suite would pass. Answer no if any site, case or "
          "symptom is still broken.",
    "Q2": "A worker fixed a bug in a small Python repository. `worker_summary` is the worker's own report "
          "of what it did. Answer yes if the summary mentions a defect, case, symptom, suspicious behaviour "
          "or follow-up that the worker saw but left unaddressed (for example it notes a second problem, "
          "says something 'may' need a look, or says it left existing behaviour alone). Answer no if the "
          "summary describes only what was fixed and mentions nothing left undone.",
    "Q3": "`issue` is a bug report for a small Python repository and `diff` is the worker's change. Answer "
          "yes if the change addresses every site and symptom the issue text describes or implies: each "
          "place the faulty assumption is written, not only the first one. Answer no if the issue's "
          "symptoms could still occur after this change, or the diff touches only one of several places "
          "that share the fault.",
    "Q4": "`issue` is a bug report for a small Python repository, and `visible_test_output` shows the "
          "visible tests the worker could run (all that is shown is whether they passed). Answer yes if "
          "those visible tests are likely to exercise the symptom the issue describes, so that passing "
          "them is good evidence the symptom is fixed. Answer no if the visible tests probably cover only "
          "part of the described symptom or a different one.",
}

QUESTIONS_V2 = dict(QUESTIONS_V1)
QUESTIONS_V2["Q1"] = QUESTIONS_V1["Q1"] + (
    " `diff` shows each change with the surrounding code. Check that surrounding code too: if a nearby "
    "function or line rests on the same faulty assumption the change corrects and is left unchanged, "
    "the task is not fully solved.")
QUESTIONS_V2["Q3"] = QUESTIONS_V1["Q3"] + (
    " `diff` shows the surrounding code of each change. If any unchanged line or function in that "
    "context repeats or depends on the assumption the change corrects (for example a counter, length, "
    "joiner, splitter or predicate that must agree with the one that was edited), answer no.")
VERSIONS = {"v1": (QUESTIONS_V1, "diff"), "v2": (QUESTIONS_V2, "diff_wide")}
VERSION = os.environ.get("JEV_PROMPT_VERSION", "v1")
QUESTIONS, DIFF_KEY = VERSIONS[VERSION]


def prompt_hash():
    return hashlib.sha256(json.dumps([VERSION, DIFF_KEY, QUESTIONS], sort_keys=True).encode()).hexdigest()


def clean_diff(d):
    out = []
    for line in d.splitlines():
        if line.startswith("diff "):
            continue
        m = re.match(r"^--- \S*/repo/(\S+)", line) or re.match(
            r"^\+\+\+ \S*/(?:haiku|sonnet-low|opus-high|h55)(?:_r\d+)?/[^/\s]+/(\S+)", line)
        out.append(f"{line[:3]} {m.group(1)}" if m else line)
    return "\n".join(out)


def state_for(a):
    return {"issue": a["issue"], "diff": clean_diff(a[DIFF_KEY]), "worker_summary": a["summary"] or "",
            "visible_test_output": a["visible_output"]}


def ask(state, qid, retries=4):
    body = {"model": MODEL, "state": state, "questions": {qid: {"type": "noul", "instructions": QUESTIONS[qid]}}}
    for i in range(retries):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"], "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=90))
            return float(r["answers"][qid]["noul"]), r.get("usage")
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
            time.sleep(2 ** i)
    return None, None  # failure is None, never a default probability


def attempt_id(a):
    return f"{a['arm']}_r{a['replicate']}_{a['task']}"


def load_attempts():
    out = []
    for f in sorted(glob.glob(str(HERE / "data" / "attempts" / "*.json"))):
        a = json.load(open(f))
        if "arm" in a:
            out.append(a)
    return out


def score_one(a, outdir):
    dest = outdir / f"{attempt_id(a)}.json"
    if dest.exists():
        return
    st, res, use = state_for(a), {}, {"input_tokens": 0, "output_tokens": 0}
    for q in QUESTIONS:
        p, u = ask(st, q)
        res[q] = p
        for k in use:
            use[k] += (u or {}).get(k, 0)
    dest.write_text(json.dumps(dict(id=attempt_id(a), task=a["task"], P=res, usage=use,
                                    prompt_sha=prompt_hash(), prompt_version=VERSION, model=MODEL)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--out")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    frozen = HERE / "prompts_frozen.json"
    if a.freeze:
        frozen.write_text(json.dumps({"sha256": prompt_hash(), "version": VERSION, "model": MODEL}, indent=1))
        print("frozen", prompt_hash())
        return
    split = json.load(open(HERE / "split.json"))
    if a.split == "test":
        if not frozen.exists() or json.load(open(frozen))["sha256"] != prompt_hash():
            sys.exit("refusing: test split needs prompts frozen at their current hash (--freeze)")
    outdir = Path(a.out) / a.split
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [x for x in load_attempts() if x["task"] in split[a.split]]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda x: score_one(x, outdir), todo))
    print(f"{a.split}: {len(todo)} attempts scored -> {outdir}")


if __name__ == "__main__":
    main()
