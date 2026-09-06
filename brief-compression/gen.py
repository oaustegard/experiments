#!/usr/bin/env python3
"""Generate the run fixtures and the three brief styles.

Usage: python3 gen.py <runs_root> [--n 20] [--seed 7]

Each run gets its own directory under <runs_root>/<style>_<model>_<i>/ with:
  input.log   -- 240 lines, some containing ERROR, some DEBUG, some both
  log.txt     -- 3 pre-existing lines (the subagent must append exactly one)
The subagent is asked to write out.json into the same directory.

briefs.json maps run id -> {style, model, brief, dir, gold}.
The three styles carry identical information; only the phrasing differs.
"""
import argparse, hashlib, json, random
from pathlib import Path

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
    def ntok(s): return len(_ENC.encode(s))
except Exception:  # tiktoken missing: degrade to None, not 0
    def ntok(s): return None

STYLES = ["prose", "telegraphic", "structured"]
MODELS = ["haiku", "sonnet"]

WORDS = ("request served cache miss retry queue worker timeout db conn pool "
         "flush user login token refresh page render index scan").split()

def make_input(rng):
    lines = []
    for i in range(240):
        w = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 7)))
        r = rng.random()
        if r < 0.18:
            tag = "ERROR"
        elif r < 0.30:
            tag = "DEBUG"
        elif r < 0.36:
            tag = "DEBUG ERROR"          # contains both -> excluded
        elif r < 0.40:
            tag = "error"                # lowercase -> not counted
        elif r < 0.43:
            tag = "WARN ERRORS"          # substring ERROR -> counted
        else:
            tag = rng.choice(["INFO", "WARN", "TRACE"])
        lines.append(f"2026-09-06T14:{i//60:02d}:{i%60:02d}Z {tag} {w}")
    return "\n".join(lines) + "\n"

def gold_count(text):
    return sum(1 for l in text.splitlines() if "ERROR" in l and "DEBUG" not in l)

def briefs(run, d):
    inp, out, log = f"{d}/input.log", f"{d}/out.json", f"{d}/log.txt"
    prose = (
        f"I need you to do a small log-audit task. The input file is at {inp}. "
        f"Please count the number of lines in that file that contain the string ERROR "
        f"(case-sensitive substring match) but do not contain the string DEBUG. "
        f"Then write a JSON file to {out} with exactly three keys: \"count\" (that number, "
        f"as an integer), \"sha256\" (the hex SHA-256 of the input file's bytes), and "
        f"\"run\" (the string \"{run}\"). After that, append exactly one line reading "
        f"\"{run} done\" to the existing log file at {log}. Please don't modify the input "
        f"file and don't create any other files. When you're finished, reply with the "
        f"single word DONE and nothing else."
    )
    telegraphic = (
        f"Logaudit {inp}. Count lines w/ERROR excl DEBUG, casesens substring. "
        f"Emit {out} exactly3keys: count int, sha256 hexofinputbytes, run \"{run}\". "
        f"Append1line \"{run} done\" existing {log}. Inputreadonly, nootherfiles. "
        f"Reply DONE only."
    )
    structured = (
        f"Task: log audit.\n"
        f"- Input: {inp} (read-only)\n"
        f"- Count lines containing ERROR and not DEBUG (case-sensitive substring)\n"
        f"- Write {out} with exactly 3 keys: count (int), sha256 (hex of input bytes), run (\"{run}\")\n"
        f"- Append exactly one line \"{run} done\" to existing {log}\n"
        f"- Create no other files\n"
        f"- Final reply: the single word DONE"
    )
    return {"prose": prose, "telegraphic": telegraphic, "structured": structured}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    root = Path(a.root)  # no resolve(): keep the caller-chosen (short) path in the brief
    rng = random.Random(a.seed)
    # one shared input per replicate index so styles/models see identical data
    inputs = [make_input(rng) for _ in range(a.n)]
    out = {}
    for i in range(a.n):
        for style in STYLES:
            for model in MODELS:
                run = "r" + hashlib.sha1(f"{a.seed}:{style}:{model}:{i}".encode()).hexdigest()[:6]
                d = root / run
                d.mkdir(parents=True, exist_ok=True)
                (d / "input.log").write_text(inputs[i])
                (d / "log.txt").write_text("boot ok\nschema ok\nready\n")
                b = briefs(run, str(d))[style]
                out[run] = {
                    "cell": f"{style}_{model}_{i:02d}", "style": style, "model": model, "i": i, "dir": str(d),
                    "brief": b, "brief_chars": len(b), "brief_o200k": ntok(b),
                    "gold": {
                        "count": gold_count(inputs[i]),
                        "sha256": hashlib.sha256(inputs[i].encode()).hexdigest(),
                    },
                }
    (root / "briefs.json").write_text(json.dumps(out, indent=1))
    by = {}
    for r in out.values():
        by.setdefault(r["style"], []).append((r["brief_chars"], r["brief_o200k"]))
    for s, v in by.items():
        print(s, "chars", sum(c for c, _ in v) / len(v), "o200k", sum(t or 0 for _, t in v) / len(v))
    print("gold counts", [gold_count(x) for x in inputs])

if __name__ == "__main__":
    main()
