#!/usr/bin/env python3
"""Gemini-drafter arm on the positive-writer-brief fixture.

Arm B (Opus 5.5 drafts from brief_b.md) and arm C (B plus the declauding editor pass) are reused from
../positive-writer-brief. This adds arm G: the same system prompt and task prompts, drafted by Gemini,
then G-final: the same Opus editor pass on G's drafts.

Stages, in order:
  gdraft  Gemini drafts, run here directly            -> drafts/G/<task>__<k>.md
  edit    Opus editor on G drafts (headless children) -> finals/G/<task>__<k>.md
  fresh   Sonnet newcomer read of G drafts and finals -> scores/fresh/<stage>__G__<task>__<k>.json
  judge   blind rating of B and G together, Opus and Sonnet (headless) -> scores/judge/
  gjudge  the same judge prompts sent to Gemini, run here directly  -> scores/judge/gemini__...

    python3 pipeline.py gdraft            # resumes: skips drafts already on disk
    python3 pipeline.py prep STAGE        # edit | fresh | judge (judge also writes runs/gjudge/tasks.jsonl)
    python3 pipeline.py collect STAGE     # copies runs/STAGE/results into the tree
    python3 pipeline.py gjudge
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import random
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from _lib.paths import experiment  # noqa: E402

import gemini  # noqa: E402

PWB = experiment("positive-writer-brief")
_spec = importlib.util.spec_from_file_location("pwb_pipeline", PWB / "pipeline.py")
pwb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pwb)

GEMINI = "gemini-4-carbon"
THINKING = "low"           # the Opus drafter and editor ran at effort low
GATEWAY_CONCURRENCY = 2    # METHODS.md: the Cloudflare gateway throttles; start at 2
TASKS = pwb.TASKS
N = pwb.N_DRAFTS
RUNS = HERE / "runs"
DRAFT_SCHEMA = {"type": "object", "properties": {"draft": {"type": "string"}}, "required": ["draft"]}
_lock = threading.Lock()


def path(stage: str, arm: str, tid: str, k: int) -> Path:
    root = PWB if arm == "B" else HERE
    return root / ("drafts" if stage == "raw" else "finals") / arm / f"{tid}__{k}.md"


def ledger(stage: str, row: dict) -> None:
    p = RUNS / stage / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, p.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _parallel(jobs, fn):
    fails = 0
    with cf.ThreadPoolExecutor(GATEWAY_CONCURRENCY) as ex:
        for ok in ex.map(fn, jobs):
            fails += not ok
    return fails


def gdraft() -> int:
    system = pwb.system_prompt("B")
    jobs = [(t, k) for t in TASKS for k in range(N) if not path("raw", "G", t["id"], k).exists()]
    print(f"{len(jobs)} drafts to write", flush=True)

    def one(job):
        t, k = job
        out, meta = gemini.generate(system, pwb.draft_prompt(t), DRAFT_SCHEMA, GEMINI, THINKING)
        ledger("gdraft", {"id": f"G__{t['id']}__{k}", "ok": out is not None, **meta})
        if out is None or not out.get("draft", "").strip():
            print(f"FAIL G__{t['id']}__{k}: {meta.get('error') or meta.get('finish')}", flush=True)
            return False
        p = path("raw", "G", t["id"], k)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out["draft"].rstrip() + "\n")
        print(f"ok G__{t['id']}__{k} {meta.get('seconds')}s", flush=True)
        return True

    fails = _parallel(jobs, one)
    print(f"gdraft done, {fails} failed")
    return 1 if fails else 0


def prep(stage: str) -> None:
    out = RUNS / stage
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if stage == "edit":
        sp = pwb.editor_system()
        for t in TASKS:
            for k in range(N):
                p = path("raw", "G", t["id"], k)
                rows.append({"id": f"G__{t['id']}__{k}", "prompt": pwb.edit_prompt(t, p),
                             "system_prompt": sp, "model": pwb.EDITOR, "tools": "",
                             "json_schema": {"type": "object",
                                             "properties": {"final": {"type": "string"},
                                                            "notes": {"type": "string"}},
                                             "required": ["final", "notes"]}})
    elif stage == "fresh":
        for st in ("raw", "final"):
            for t in TASKS:
                for k in range(N):
                    p = path(st, "G", t["id"], k)
                    rows.append({"id": f"{st}__G__{t['id']}__{k}", "model": pwb.READER, "tools": "",
                                 "prompt": pwb.FRESH_PROMPT.format(audience=t["audience"], surface=t["surface"],
                                                                   text=p.read_text()),
                                 "system_prompt": "You are a careful first-time reader.",
                                 "json_schema": pwb.FRESH_SCHEMA})
    elif stage == "judge":
        rng = random.Random(20261117)
        key, grows = {}, []
        for st in ("raw", "final"):
            for t in TASKS:
                items = [(arm, k, path(st, arm, t["id"], k)) for arm in ("B", "G") for k in range(N)]
                missing = [str(p) for _, _, p in items if not p.exists()]
                if missing:
                    raise SystemExit(f"missing texts: {missing[:3]}")
                rng.shuffle(items)
                ids = rng.sample(range(100, 1000), len(items))
                block = ""
                for (arm, k, p), i in zip(items, ids):
                    key[f"{st}__{t['id']}__D{i}"] = {"arm": arm, "k": k}
                    block += f"\n=== draft D{i} ===\n\n{p.read_text()}\n"
                prompt = pwb.task_block(t) + "\n" + block
                for jname, jmodel in pwb.JUDGES.items():
                    rows.append({"id": f"{jname}__{st}__{t['id']}", "model": jmodel, "tools": "",
                                 "system_prompt": pwb.JUDGE_SYSTEM, "json_schema": pwb.JUDGE_SCHEMA,
                                 "prompt": prompt})
                grows.append({"id": f"gemini__{st}__{t['id']}", "prompt": prompt})
        (HERE / "scores").mkdir(exist_ok=True)
        (HERE / "scores" / "judge_key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
        g = RUNS / "gjudge"
        g.mkdir(parents=True, exist_ok=True)
        with (g / "tasks.jsonl").open("w") as f:
            for r in grows:
                f.write(json.dumps(r) + "\n")
    else:
        raise SystemExit(f"unknown stage {stage}")
    with (out / "tasks.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} tasks -> {out / 'tasks.jsonl'}")


def gjudge() -> int:
    rows = [json.loads(line) for line in (RUNS / "gjudge" / "tasks.jsonl").read_text().splitlines()]
    dst = HERE / "scores" / "judge"
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [r for r in rows if not (dst / f"{r['id']}.json").exists()]

    def one(r):
        out, meta = gemini.generate(pwb.JUDGE_SYSTEM, r["prompt"], pwb.JUDGE_SCHEMA, GEMINI, THINKING)
        ledger("gjudge", {"id": r["id"], "ok": out is not None, **meta})
        if out is None or len(out.get("ratings", [])) != 2 * N:
            print(f"FAIL {r['id']}: {meta.get('error') or len((out or {}).get('ratings', []))}", flush=True)
            return False
        (dst / f"{r['id']}.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"ok {r['id']}", flush=True)
        return True

    fails = _parallel(jobs, one)
    print(f"gjudge done, {fails} failed")
    return 1 if fails else 0


def collect(stage: str) -> None:
    n = 0
    for f in sorted((RUNS / stage / "results").glob("*.json")):
        so = json.loads(f.read_text()).get("structured_output")
        if so is None:
            continue
        parts = f.stem.split("__")
        if stage == "edit":
            arm, tid, k = parts
            dst = HERE / "finals" / arm / f"{tid}__{k}.md"
            notes = HERE / "finals" / "notes"
            notes.mkdir(parents=True, exist_ok=True)
            (notes / f"{arm}__{tid}__{k}.txt").write_text(so.get("notes", "") + "\n")
            text = so["final"]
        else:
            dst = HERE / "scores" / stage / f"{f.stem}.json"
            text = json.dumps(so, indent=1, ensure_ascii=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text.rstrip() + "\n")
        n += 1
    print(f"collected {n} -> {stage}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gdraft":
        sys.exit(gdraft())
    if cmd == "gjudge":
        sys.exit(gjudge())
    {"prep": prep, "collect": collect}[cmd](sys.argv[2])
