"""The dry runs of dryrun.py, on Oskar's Mac GPU through the GPU Bridge, with the exact model and
sampler the chat page ships: 8-bit export, batch 1, the page's sampling arithmetic (seeded here).

    python3 bridge_dry.py submit --model A8p_q8 --variants S0,S1 --decode t0.7r --seed 0 --tag r2
    python3 bridge_dry.py collect --tag r2 --out runs/r2.jsonl

Jobs are named mc-<tag>-<model>-<decode>-s<seed>-<k>; each holds up to --per prompts. Rows match
dryrun.py's, plus "src": "mac".
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

MWE = experiment("monad-weight-edit")
sys.path.insert(0, str(MWE))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from bridge import GB, model_inputs, tok  # noqa: E402
from common import END, loop_period, split_think  # noqa: E402
from dryrun import DECODES, build  # noqa: E402

NAME = {"A8p_q8": "A8p", "base_q8": "monad"}


def submit(a):
    variants = json.loads((HERE / "variants.json").read_text())
    probes = json.loads((HERE / a.probe_file).read_text())
    if a.probes:
        keep = set(a.probes.split(","))
        probes = [p for p in probes if p["id"] in keep]
    t = tok()
    items = [(f"{v}|{p['id']}", t.encode(build(variants[v], p), add_special_tokens=False))
             for v in a.variants.split(",") for p in probes]
    dec = DECODES[a.decode]
    for k in range(0, len(items), a.per):
        chunk = items[k:k + a.per]
        jid = f"mc-{a.tag}-{a.model}-{a.decode}-s{a.seed}-{k // a.per}"
        d = MWE / "bridge/jobs" / jid
        d.mkdir(parents=True, exist_ok=True)
        spec, _ = model_inputs(MWE / "export" / a.model, d, weights=not (a.no_weights or k > 0 and a.reuse))
        cfg = {"mode": "gen", "budgetMs": 520000, "cacheTag": "mc1",
               "batches": [{"uids": [u], "prompts": [ids], "maxNew": a.max_new} for u, ids in chunk]}
        if dec:
            cfg["sample"] = dict(dec, seed=a.seed * 1000 + k + 1)
        cmd = GB + ["submit", str(MWE / "bridge/job.js"), "--id", jid, "--note", f"monad-chat dry run {a.tag}",
                    "--timeout", "600000", "--input", f"cfg=@{json.dumps(cfg)}"]
        for s in spec:
            cmd += ["--input", s]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(jid, (r.stdout + r.stderr).strip()[-200:])


def collect(a):
    clone = Path("/home/user/claude-workspace/.spokes/gpu-bridge-comms")
    subprocess.run(["git", "-C", str(clone), "pull", "--rebase", "--quiet", "origin", "main"], check=False)
    ids = sorted(p.name for p in (clone / "jobs").iterdir() if p.name.startswith(f"mc-{a.tag}-"))
    t = tok()
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, pending = [], []
    for jid in ids:
        st = subprocess.run(GB + ["status", jid], capture_output=True, text=True)
        if st.returncode != 0:
            pending.append(jid)
            continue
        d = MWE / "bridge/results" / jid
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(GB + ["fetch", jid, "--out", str(d)], capture_output=True, text=True)
        res = json.loads(r.stdout)
        o = res["outputs"]
        model, decode, seed = jid.split("-")[2], jid.split("-")[3], int(jid.split("-")[4][1:])
        toks = np.load(d / "tokens.npy") if o["lengths"]["length"] else np.zeros(0, np.int32)
        lens, stopped = np.load(d / "lengths.npy"), np.load(d / "stopped.npy")
        ms = {b["i"]: b for b in o["batches"]}
        pos = 0
        for i, (uid, n, s) in enumerate(zip(o["uids"], lens.tolist(), stopped.tolist())):
            ids_ = toks[pos:pos + n].tolist()
            pos += n
            text = t.decode(ids_)
            think, ans, closed = split_think(text)
            v, p = uid.split("|")
            b = ms.get(i, {})
            rows.append({"model": NAME[model], "variant": v, "decode": decode, "seed": seed, "probe": p, "stopped": bool(s),
                         "closed": closed, "loop": (not s) and loop_period(ids_) is not None, "tokens": n, "answer": ans,
                         "think": think, "src": "mac", "step_ms": b.get("stepMedMs")})
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"rows": len(rows), "pending": pending}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect"])
    ap.add_argument("--model", default="A8p_q8")
    ap.add_argument("--variants", default="S0")
    ap.add_argument("--probes", default="")
    ap.add_argument("--probe-file", default="probes.json")
    ap.add_argument("--decode", default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per", type=int, default=28)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-weights", action="store_true", help="the Mac tab already holds this model's session (cacheTag mc1)")
    ap.add_argument("--reuse", action="store_true", help="send weights with the first job only")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"submit": submit, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
