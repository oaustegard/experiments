"""Relay context management on Needle Retention: one stateless call per chunk, state carried in files.

Conditions
  state  a headless Claude Code subagent (`claude -p`, Read/Edit/Write tools) per chunk sees the
         chunk and maintains state.md, its only memory across chunks. Model-authored retention.
  jev    no model. Jev scores every line of the chunk against the task; lines at p >= 0.5 are
         appended verbatim to kept.md. Extractive retention by an outside scorer.
  both   Jev runs first and appends to kept.md; the subagent is shown what Jev kept and maintains
         state.md for anything Jev missed. Graded on the union.

All conditions share one cap on what is carried forward (CAP_TOKENS, o200k). Over the cap, `jev`
drops its lowest-scored lines; a subagent's state.md is cut at the cap and the event is logged.

Resumable: every episode checkpoints after each chunk under work/, and a finished episode writes
results/<cond>__<variant>__<n>__<seed>.json. The run ends by printing `done -> results/summary.json`.

    python3 relay.py --conds state,jev,both --variants unlabeled --sizes 8,32,96 --seeds 0,1 --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import tiktoken

from gen import TASK, episode

HERE = Path(__file__).resolve().parent
WORK, RES = HERE / "work", HERE / "results"
CAP_TOKENS = 24_000          # 32k limit - one ~4k chunk - 2k response reserve - instructions
MODEL = "claude-haiku-4-5-20251001"
JEV_RPM = 45                 # gateway caps Jev at 50/min (METHODS.md); stay under it
JEV_KEEP = 0.5
ENC = tiktoken.get_encoding("o200k_base")


def ntok(s: str) -> int:
    return len(ENC.encode(s))


# ---------------------------------------------------------------- Jev
_jev_lock, _jev_last = threading.Lock(), [0.0]


def jev_scores(task: str, chunk: str) -> tuple[dict[str, float], int]:
    """P(line needed) for every [id#hash] line of a chunk, one gateway call."""
    lines = [l for l in chunk.splitlines() if l.startswith("[")]
    state = {"task": task, "lines": {f"l{i:03d}": l for i, l in enumerate(lines)}}
    qs = {k: {"type": "noul", "instructions": f"Does `lines.{k}` contain information that an assistant "
                                              f"given `task` would need to keep?"} for k in state["lines"]}
    body = json.dumps({"model": "typesafe/jev", "input": {"state": state, "questions": qs}}).encode()
    url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}/ai/run"
    hdr = {"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}", "Content-Type": "application/json",
           "cf-aig-gateway-id": os.environ["CF_GATEWAY_ID"], "cf-aig-skip-cache": "true"}
    last = None
    for attempt in range(8):
        with _jev_lock:  # client-side pacer: a fixed window punishes backoff (METHODS.md)
            wait = _jev_last[0] + 60 / JEV_RPM - time.time()
            if wait > 0:
                time.sleep(wait)
            _jev_last[0] = time.time()
        try:
            req = urllib.request.Request(url, data=body, headers=hdr, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                res = json.load(r)["result"]["result"]
            ans = {state["lines"][k]: a["noul"] for k, a in res["answers"].items()}
            return ans, res.get("usage", {}).get("input_tokens", 0)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
            time.sleep(min(30, 2 * 2 ** attempt))
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(30, 2 * 2 ** attempt))
    raise RuntimeError(f"Jev failed: {last}")


def pack(kept: list[list], cap: int) -> list[list]:
    """kept = [[line, score], ...]; drop lowest-scored lines until the text fits the cap."""
    while kept and ntok("\n".join(l for l, _ in kept)) > cap:
        kept.remove(min(kept, key=lambda x: x[1]))
    return kept


# ---------------------------------------------------------------- subagent
SYSTEM = ("You are one step in a relay. Each step is a fresh process: you see one chunk of a stream, "
          "and the files in your working directory are the only thing that survives to the next step. "
          "Use the Read, Edit and Write tools on files in the current directory. Reply DONE when finished.")


def subagent(task: str, chunk: str, wd: Path, cap_note: str, jev_note: str) -> dict:
    prompt = (f"TASK (same for every step):\n{task}\n\n"
              f"Your memory is ./state.md. It is the only text carried to the next step; you will never see "
              f"this chunk again. {cap_note}\n{jev_note}\n"
              f"Update ./state.md now for this chunk, then reply DONE.\n\n"
              f"<<<CHUNK BEGIN>>>\n{chunk}\n<<<CHUNK END>>>")
    cmd = ["claude", "-p", "--model", MODEL, "--tools", "Read", "Edit", "Write",
           "--permission-mode", "acceptEdits", "--setting-sources", "", "--strict-mcp-config",
           "--no-session-persistence", "--system-prompt", SYSTEM, "--output-format", "json"]
    for attempt in range(3):
        p = subprocess.run(cmd, input=prompt, cwd=wd, capture_output=True, text=True, timeout=900)
        try:
            d = json.loads(p.stdout)
            mu = d.get("modelUsage", {}).get(MODEL, {})
            return {"ok": not d.get("is_error"), "turns": d.get("num_turns"), "cost": d.get("total_cost_usd", 0),
                    "in": mu.get("inputTokens", 0), "out": mu.get("outputTokens", 0),
                    "result": (d.get("result") or "")[:200]}
        except json.JSONDecodeError:
            err = (p.stderr or p.stdout)[-300:]
            time.sleep(10 * (attempt + 1))
    return {"ok": False, "error": err, "cost": 0, "in": 0, "out": 0}


# ---------------------------------------------------------------- episode loop
def run_episode(cond: str, variant: str, n: int, seed: int, log) -> Path:
    out = RES / f"{cond}__{variant}__{n}__{seed}.json"
    if out.exists():
        return out
    ep = episode(n, seed, variant)
    task = TASK[variant]
    wd = WORK / f"{cond}__{variant}__{n}__{seed}"
    wd.mkdir(parents=True, exist_ok=True)
    prog_f = wd / "progress.json"
    prog = json.loads(prog_f.read_text()) if prog_f.exists() else {"next": 0, "steps": [], "kept": []}
    (wd / "state.md").touch()
    for c in range(prog["next"], n):
        chunk, step = ep["chunks"][c], {"chunk": c}
        if cond in ("jev", "both"):
            sc, jt = jev_scores(task, chunk)
            new = [[l, s] for l, s in sc.items() if s >= JEV_KEEP]
            cap = CAP_TOKENS if cond == "jev" else CAP_TOKENS - ntok((wd / "state.md").read_text())
            prog["kept"] = pack(prog["kept"] + new, max(cap, 0))
            (wd / "kept.md").write_text("\n".join(l for l, _ in prog["kept"]))
            step.update(jev_new=len(new), jev_tokens=jt)
        if cond in ("state", "both"):
            used = ntok((wd / "state.md").read_text())
            room = CAP_TOKENS - (ntok((wd / "kept.md").read_text()) if cond == "both" else 0)
            cap_note = (f"state.md may hold at most {room} tokens; it holds {used} now. Anything past the "
                        f"limit is cut off.")
            jev_note = ""
            if cond == "both":
                jl = [l for l, s in sc.items() if s >= JEV_KEEP]
                jev_note = ("A filter has already saved these lines from this chunk verbatim to a separate "
                            "store that is graded with state.md (do not copy them again):\n"
                            + ("\n".join(jl) if jl else "(none)")
                            + "\nUse state.md only for lines the task requires that the filter missed.")
            r = subagent(task, chunk, wd, cap_note, jev_note)
            s = (wd / "state.md").read_text()
            if ntok(s) > room:
                toks = ENC.encode(s)[:max(room, 0)]
                (wd / "state.md").write_text(ENC.decode(toks))
                r["truncated_from"] = len(ENC.encode(s))
            step.update(agent=r)
        step["carried_tokens"] = ntok((wd / "state.md").read_text()) + (
            ntok((wd / "kept.md").read_text()) if (wd / "kept.md").exists() else 0)
        prog["steps"].append(step)
        prog["next"] = c + 1
        prog_f.write_text(json.dumps(prog))
        log(f"{cond} {variant} n={n} s={seed} chunk {c + 1}/{n} carried={step['carried_tokens']}")
    final = (wd / "state.md").read_text() + "\n" + ((wd / "kept.md").read_text() if (wd / "kept.md").exists() else "")
    out.write_text(json.dumps({"cond": cond, "variant": variant, "n": n, "seed": seed, "model": MODEL,
                               "final": final, "steps": prog["steps"]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="state,jev,both")
    ap.add_argument("--variants", default="unlabeled")
    ap.add_argument("--sizes", default="8,32,96")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    lock = threading.Lock()

    def log(msg: str) -> None:
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    jobs = [(c, v, int(n), int(s)) for v in a.variants.split(",") for n in a.sizes.split(",")
            for s in a.seeds.split(",") for c in a.conds.split(",")]
    jobs.sort(key=lambda j: -j[2])  # longest first so the tail is short
    failed = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_episode, *j, log): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                log(f"finished {futs[f]} -> {f.result().name}")
            except Exception as e:  # keep the other episodes going; the job reruns on resume
                failed += 1
                log(f"FAILED {futs[f]}: {type(e).__name__}: {e}")
    import grade
    grade.main()
    print(f"incomplete: {failed} episodes failed; rerun to resume" if failed
          else "done -> results/summary.json", flush=True)


if __name__ == "__main__":
    main()
