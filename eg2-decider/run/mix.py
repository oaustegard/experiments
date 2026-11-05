"""A3's training mix: a source-balanced sample of tasksource/tasksource-jev-typed-decisions.

    python3 -I run/mix.py sample [--n 8000] [--per-source 12]     # writes out/ts/mix.jsonl
    python3 -I run/mix.py embed --state-prompt P --opt-prompt P --template T

Rows are one question each (state, question, options, target). Typed Decisions rows are dropped (none
were found in shard 0, checked by source name and by TD state keys). noul rows carry one target
p(true) and no options; they become true/false with the benchmark's default descriptions. Option
strings like "routine: Routine." split into name and description; bare labels keep the label as both.
States are capped at 512 tokens when embedded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "browser-decider" / "run"))
from common import NOUL_DEFAULT  # noqa: E402

TS = ROOT / "out" / "ts"
MIX = TS / "mix.jsonl"
SHARD = TS / "train0.parquet"  # data/train-00000-of-00006.parquet, 166,639 rows, 749 sources


def to_example(r):
    kind = r["kind"]
    if kind == "noul":
        p = float(r["target"][0])
        opts, gold = list(NOUL_DEFAULT.items()), [p, 1 - p]
    else:
        opts = []
        for o in r["options"]:
            name, sep, desc = str(o).partition(": ")
            opts.append((name, desc) if sep and desc else (str(o), str(o)))
        gold = [float(x) for x in r["target"]]
    return {"case": r["group_id"], "workflow": r["source"], "qname": r["question_id"], "type": kind,
            "question": f"{kind}: {r['question']}", "instructions": r["question"], "options": opts,
            "state_text": r["state"], "gold": gold}


def sample(n, per_source, seed=0):
    import pandas as pd
    d = pd.read_parquet(SHARD)
    d = d[d.state.str.len() > 0]
    d = d[~d.source.str.contains("LocalLLaMA|typed-decisions/(customer|invoice|agent|security)", regex=True)]
    nopt = d.options.map(len)
    d = d[((d.kind == "noul") & (nopt == 0)) | ((d.kind != "noul") & nopt.between(2, 16))]
    rng = np.random.default_rng(seed)
    d = d.sample(frac=1.0, random_state=seed)
    d = d.groupby("source", group_keys=False).head(per_source)
    if len(d) > n:
        d = d.sample(n, random_state=seed)
    rows = [to_example(r) for r in d.to_dict("records")]
    rows = [e for e in rows if abs(sum(e["gold"]) - 1) < 1e-3]
    rng.shuffle(rows)
    MIX.write_text("\n".join(json.dumps(e) for e in rows))
    kinds = {}
    for e in rows:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print(f"{len(rows)} rows, {len({e['workflow'] for e in rows})} sources, {kinds}")


def load():
    return [json.loads(l) for l in MIX.read_text().splitlines()]


def embed(state_prompt, opt_prompt, template):
    import arms
    from enc import EG2, encoder
    rows = load()
    m = encoder(tag="mix")
    t0 = time.time()
    m.embed([e["question"] for e in rows], prompt=opt_prompt, max_tokens=256, log_every=1000)
    flat = [arms.opt_text(e, n, d, template) for e in rows for n, d in e["options"]]
    m.embed(flat, prompt=opt_prompt, max_tokens=256, log_every=5000)
    print(f"questions+options done {time.time() - t0:.0f}s", flush=True)
    m.embed([e["state_text"] for e in rows], prompt=state_prompt, max_tokens=512, log_every=500)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--per-source", type=int, default=12)
    ap.add_argument("--state-prompt")
    ap.add_argument("--opt-prompt")
    ap.add_argument("--template")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.n, a.per_source)
    else:
        embed(a.state_prompt, a.opt_prompt, a.template)
