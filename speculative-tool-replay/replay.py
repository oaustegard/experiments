"""Ask Jev, at every prediction point, which read-only call the agent makes next.

One Choice per point: options = up to 60 most-recent candidates + "none". State = the last human
message and a condensed tail of the session before the point, secrets scrubbed. Results append to
/home/user/spec-data/replay.jsonl (checkpoint; re-running skips done points). Transcript text stays
out of the repo.
"""
import glob, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "jev-tag-encoder"), str(HERE.parent / "subagent-context-filter")]
import jev  # noqa: E402  (transport, pacing, WAF detection)
from chunking import scrub  # noqa: E402
from parse import points  # noqa: E402
from candidates import candidates  # noqa: E402

OUT = Path("/home/user/spec-data/replay.jsonl")
TX = "/home/user/spec-data/tx/*.jsonl"
N_CAND, STATE_CHARS = 60, 12000
INSTR = ("The state is the most recent part of a coding agent's session, oldest first. Which one of the "
         "listed tool calls will the agent make in its very next message? Choose none if its next message "
         "makes none of these calls.")


def render(e):
    if e["kind"] == "user":
        return ("[harness] " if e.get("meta") else "[user] ") + e["text"][:1500]
    if e["kind"] == "text":
        return "[agent] " + e["text"][:1500]
    if e["kind"] == "call":
        return "[call] " + e["name"] + " " + json.dumps(e["input"])[:600]
    return "[result] " + e["text"][:600]


def state(ev, idx):
    human = next((e["text"][:2000] for e in reversed(ev[: idx + 1]) if e["kind"] == "user" and not e.get("meta")), "")
    tail, n = [], 0
    for e in reversed(ev[: idx + 1]):
        s = render(e)
        if n + len(s) > STATE_CHARS:
            break
        tail.append(s); n += len(s)
    txt, _ = scrub("\n".join(reversed(tail)))
    human, _ = scrub(human)
    return {"task": human, "recent_session": txt}


def main():
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pid"] for l in OUT.read_text().splitlines() if l.strip()}
    todo = []
    for f in sorted(glob.glob(TX)):
        ev, ps = points(f)
        for p in ps:
            pid = f"{p['session'][:26]}:{p['idx']}"
            if pid not in done:
                todo.append((pid, ev, p))
    print(f"{len(todo)} points to run, {len(done)} done", flush=True)
    lock, n = threading.Lock(), [0]

    def one(item):
        pid, ev, p = item
        cands = candidates(ev, p["idx"], limit=N_CAND)
        crit = {f"c{i:02d}": scrub(c[:300])[0] for i, c in enumerate(cands)}
        crit["none"] = "None of the listed calls: the agent writes text, edits, runs another command, or searches"
        q = {"next": {"type": "choice", "instructions": INSTR, "criteria": crit}}
        rec = {"pid": pid, "cands": cands, "labels": p["labels"], "window": p["window"], "n_calls": p["n_calls"]}
        try:
            res, dt, _ = jev.post(state(ev, p["idx"]), q)
            a = res["answers"]["next"]
            rec.update(probs=a["probabilities"], conf=a.get("confidence"), latency=dt,
                       in_tok=res.get("usage", {}).get("input_tokens"))
        except jev.Blocked:
            rec["error"] = "blocked"
        except RuntimeError as e:
            rec["error"] = str(e)[:200]
        with lock:
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n[0] += 1
            if n[0] % 100 == 0:
                print(f"{n[0]}/{len(todo)}", flush=True)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo[:limit]))
    print("done", flush=True)


if __name__ == "__main__":
    main()
