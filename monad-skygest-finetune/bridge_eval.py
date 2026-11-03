"""The eval prompts on Oskar's Mac GPU through the GPU Bridge: the page's own runtime (onnxruntime-web
1.30, WebGPU), batch 1 like the chat page, greedy. Reuses monad-weight-edit/bridge/job.js (`gen` mode).

    python3 bridge_eval.py submit export/r2_A8p_q8 --name A8p_q8 --parts 4
    python3 bridge_eval.py collect --name A8p_q8          # -> runs/A8p_q8__bridge.jsonl, lists missing prompts

Batch 1: the int8 kernel decodes at 6.6 ms/token at batch 1 and 54 ms/step at batch 8 on this Mac
(monad-layer-parallel), so single prompts finish sooner and each stops on its own.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

MWE = experiment("monad-weight-edit")
sys.path.insert(0, str(MWE))
from common import END, loop_period, split_think  # noqa: E402
from evalgen import Tok, prompt  # noqa: E402

GB = [sys.executable, "/home/user/claude-workspace/scripts/gpu_bridge.py"]
CLONE = Path("/home/user/claude-workspace/.spokes/gpu-bridge-comms")


def items():
    return [json.loads(l) for l in open(HERE / "data/eval_prompts.jsonl")]


def submit(a):
    mdir = HERE / a.model
    tok = Tok(mdir / "tokenizer.json")
    pieces = HERE / "export/pieces.json"
    if not pieces.exists():
        pieces.write_text(json.dumps([tok.decode([i]) for i in range(8192)]))
    meta = json.loads((mdir / "meta.json").read_text())
    todo = items()
    if a.only:
        keep = set(json.loads(Path(a.only).read_text()))
        todo = [x for x in todo if x["id"] in keep]
    for k in range(a.parts):
        part = todo[k::a.parts]
        cfg = {"mode": "gen", "budgetMs": a.timeout - 90000,
               "batches": [{"uids": [x["id"]], "prompts": [tok.encode(prompt(x["q"]))], "maxNew": x["max_new"]} for x in part],
               "batchIndex": list(range(len(part)))}
        jid = f"skygest-{a.name}-{a.tag}{k}"
        cmd = GB + ["submit", str(MWE / "bridge/job.js"), "--id", jid, "--note", f"monad-skygest {a.name} part {k}",
                    "--timeout", str(a.timeout), "--target", "Mac", "--input", f"cfg=@{json.dumps(cfg)}",
                    "--input", f"meta={mdir / 'meta.json'}:u8", "--input", f"pieces={pieces}:u8"]
        if k == 0 or not a.cached:
            cmd += ["--input", f"graph={mdir / 'model.onnx'}:u8"]
            cmd += [x for i, p in enumerate(meta["parts"]) for x in ("--input", f"p{i}={mdir / p['file']}:u8")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(jid, r.returncode, (r.stdout + r.stderr).strip()[-300:])


def collect(a):
    subprocess.run(["git", "-C", str(CLONE), "pull", "--rebase", "--quiet", "origin", "main"], check=False)
    tok = Tok(HERE / "export/r2_A8p_q8/tokenizer.json")
    ev = {x["id"]: x for x in items()}
    recs, timing = {}, []
    for jd in sorted((CLONE / "jobs").glob(f"skygest-{a.name}-*")):
        jid = jd.name
        if subprocess.run(GB + ["status", jid], capture_output=True, text=True).returncode != 0:
            print(jid, "not done")
            continue
        d = HERE / "logs/bridge" / jid
        d.mkdir(parents=True, exist_ok=True)
        res = json.loads(subprocess.run(GB + ["fetch", jid, "--out", str(d)], capture_output=True, text=True).stdout)
        o = res.get("outputs") or {}
        if not o.get("uids"):
            print(jid, "error", str(res.get("error"))[:300])
            continue
        toks = np.load(d / "tokens.npy") if (d / "tokens.npy").exists() else np.zeros(0, np.int32)
        pos = 0
        for uid, n, s in zip(o["uids"], np.load(d / "lengths.npy").tolist(), np.load(d / "stopped.npy").tolist()):
            ids = toks[pos:pos + n].tolist()
            pos += n
            text = tok.decode(ids)
            text = text.split(END, 1)[0] if s else text
            trace, ans, closed = split_think(text)
            x = ev[uid]
            r = {"set": x["set"], "id": uid, "stopped": bool(s), "closed": closed, "tokens": n,
                 "loop": (not s) and loop_period(ids) is not None, "answer": ans, "trace": trace}
            if "task" in x:
                r["task"] = x["task"]
            recs[uid] = r
        timing += [b | {"job": jid, "device": res.get("device"), "tabName": res.get("tabName")} for b in o["batches"]]
    out = HERE / "runs" / f"{a.name}__bridge.jsonl"
    with open(out, "w") as f:
        for uid in ev:
            if uid in recs:
                f.write(json.dumps(recs[uid], ensure_ascii=False) + "\n")
    missing = [u for u in ev if u not in recs]
    (HERE / f"logs/missing_{a.name}.json").write_text(json.dumps(missing))
    tsum = {"tokens": sum(b["tokens"] for b in timing), "ms": sum(b["ms"] for b in timing),
            "step_med_ms": float(np.median([b["stepMedMs"] for b in timing if b.get("stepMedMs")])) if timing else None,
            "devices": sorted({str(b["device"]) for b in timing})}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / f"results/bridge_timing_{a.name}.json").write_text(json.dumps(tsum, indent=1))
    print(json.dumps({"written": len(recs), "missing": len(missing), **tsum}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600000)
    ap.add_argument("--tag", default="p")
    ap.add_argument("--only", help="JSON list of prompt ids (resubmit missing ones)")
    ap.add_argument("--cached", action="store_true", help="send weights with part 0 only (tab cache)")
    a = ap.parse_args()
    submit(a) if a.cmd == "submit" else collect(a)


if __name__ == "__main__":
    main()
