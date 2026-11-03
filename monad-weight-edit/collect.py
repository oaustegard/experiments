"""Fetch a model's bridge `gen` jobs and write runs/<name>__gpu_<split>.jsonl in run_eval.py's format,
so summarize.py scores CPU and GPU runs the same way. Also writes the jobs' timing to
results/gpu_timing_<name>.json and lists batches no job has returned yet.

    python3 collect.py base [--split test] [--prefix mwe-gen-base-]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import END, loop_period, split_think  # noqa: E402
from bridge import GB, gen_batches, tok  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prefix")
    a = ap.parse_args()
    prefix = a.prefix or f"mwe-gen-{a.name}-"
    clone = Path("/home/user/claude-workspace/.spokes/gpu-bridge-comms")
    subprocess.run(["git", "-C", str(clone), "pull", "--rebase", "--quiet", "origin", "main"], check=False)
    ids = sorted(p.name for p in (clone / "jobs").iterdir() if p.name.startswith(prefix)
                 and (("-dev-" in p.name) == (a.split == "dev")))
    t = tok()
    recs, timing, got = {}, [], set()
    for jid in ids:
        st = subprocess.run(GB + ["status", jid], capture_output=True, text=True)
        if st.returncode != 0:
            print(jid, "not done:", st.stdout.strip()[:200])
            continue
        d = HERE / "bridge/results" / jid
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(GB + ["fetch", jid, "--out", str(d)], capture_output=True, text=True)
        res = json.loads(r.stdout)
        o = res["outputs"]
        cfg = json.loads((clone / "jobs" / jid / "job.json").read_text())["inputs"]["cfg"]["json"]
        toks = np.load(d / "tokens.npy") if o["lengths"]["length"] else np.zeros(0, np.int32)
        lens = np.load(d / "lengths.npy")
        stopped = np.load(d / "stopped.npy")
        pos = 0
        for uid, n, s in zip(o["uids"], lens.tolist(), stopped.tolist()):
            ids_ = toks[pos:pos + n].tolist()
            pos += n
            text = t.decode(ids_)
            _, _, closed = split_think(text)
            recs[uid] = {"uid": uid, "task": uid.split(":")[0], "n_tok": n, "stopped": bool(s), "closed": closed,
                         "loop": None if s else loop_period(ids_), "text": text.split(END, 1)[0] if s else text}
        bidx = cfg["batchIndex"]
        for b in o["batches"]:
            got.add(bidx[b["i"]])
            timing.append(b | {"job": jid, "batch": bidx[b["i"]], "device": res.get("device"), "tabName": res.get("tabName")})
    out = HERE / "runs" / f"{a.name}__gpu_{a.split}.jsonl"
    order = [u for bt in gen_batches(a.split) for u in bt["uids"]]
    with open(out, "w") as f:
        for u in order:
            if u in recs:
                f.write(json.dumps(recs[u]) + "\n")
    missing = [i for i in range(len(gen_batches(a.split))) if i not in got]
    tsum = {"batches": timing, "missing_batches": missing,
            "tokens": int(sum(b["tokens"] for b in timing)), "ms": sum(b["ms"] for b in timing),
            "step_med_ms_median": float(np.median([b["stepMedMs"] for b in timing])) if timing else None}
    if timing:
        tsum["tok_per_s"] = tsum["tokens"] / (tsum["ms"] / 1000)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"gpu_timing_{a.name}.json").write_text(json.dumps(tsum, indent=1))
    print(json.dumps({"written": len(recs), "file": str(out), "missing_batches": missing,
                      "tok_per_s": tsum.get("tok_per_s"), "step_med_ms": tsum["step_med_ms_median"]}))


if __name__ == "__main__":
    main()
