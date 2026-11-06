"""Score the ±31 refits against the shipped spikes on the ±63 base.

    python3 -I run/refit31_score.py DIR [DIR ...]      # DIRs: fetched refit31-<set> bridge results

Each DIR holds result.json from run/refit31_job.js and bin_<id>.bin per spike (scripts/gpu_bridge.py
fetch). Writes out/base31/spikes/<set>/<id>.{json,bin} and scores each on the ±31 base (out/base31)
through LoraRef on every test case, the same way run/base31.py scored the shipped spikes, then writes
results/refit31.json with ±63 shipped, ±31 shipped and ±31 refit accuracy per spike. Rule: the ±31 base
ships only if every refit spike is within 0.005 of its shipped spike on ±63.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import stock_path  # noqa: E402
from pack_system import LoraRef, everyday_cases  # noqa: E402
from shared import BD, ROOT, examples, load_cases, score_examples  # noqa: E402
sys.path.insert(0, str(BD / "run"))
from train_enc import Enc  # noqa: E402

RULE = 0.005


def score(ref, tok, sm, sb, cases):
    ref.use(sm, sb)
    te, probs = examples(cases), []
    for e in te:
        ids, pos = tok.build(e)
        start = ids.index(tok.sep, pos[-1]) + 1
        z, _ = ref.run(ids, pos, (start, len(ids) - 1))
        T = sm["temps"].get(e["type"], 1.0)
        p = np.exp(z / T - (z / T).max())
        probs.append((p / p.sum()).tolist())
    s = score_examples(te, probs)[0]
    return {"n": len(te), "accuracy": s["accuracy"], "kl": s["kl_from_gold"]}


def main():
    b31 = ROOT / "out" / "base31"
    man = json.loads((b31 / "manifest.json").read_text())
    ref = LoraRef(man, (b31 / "weights.bin").read_bytes())
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    prior = json.loads((ROOT / "results" / "base31.json").read_text())
    out = ROOT / "results" / "refit31.json"   # each run adds to what earlier runs scored
    res = json.loads(out.read_text()) if out.exists() else {"rule": f"ship ±31 only if every refit is within {RULE} of its shipped spike on ±63", "spikes": {}}
    for d in map(Path, sys.argv[1:]):
        r = json.loads((d / "result.json").read_text())
        o = r.get("outputs", r)
        set_ = o["set"]
        res.setdefault("forward_check", {})[set_] = o["forward_check"]
        res.setdefault("device", o.get("describe"))
        (b31 / "spikes" / set_).mkdir(parents=True, exist_ok=True)
        for sid, x in o["spikes"].items():
            f = d / f"bin_{sid}.bin"
            sb = f.read_bytes() if f.exists() else np.load(d / f"bin_{sid}.npy").astype(np.uint8).tobytes()   # gpu_bridge.py fetch writes .npy
            (b31 / "spikes" / set_ / f"{sid}.json").write_text(json.dumps(x["sm"]))
            (b31 / "spikes" / set_ / f"{sid}.bin").write_bytes(sb)
            cases = load_cases(sid, "test") if set_ == "workflows" else everyday_cases(sid, "test")
            s = score(ref, tok, x["sm"], sb, cases)
            old = prior["sets"][set_]["spikes"][sid]
            rec = {"set": set_, "L63_shipped": old["L63"]["accuracy"], "L31_shipped": old["L31"]["accuracy"], "L31_refit": s["accuracy"],
                   "kl": {"L63_shipped": old["L63"]["kl"], "L31_refit": s["kl"]}, "n": s["n"], "best_epoch": x["bestEpoch"], "epochs": x["epochs"]}
            rec["delta_vs_L63"] = rec["L31_refit"] - rec["L63_shipped"]
            res["spikes"][sid] = rec
            print(f"{set_} {sid}: ±63 {rec['L63_shipped']:.4f}  ±31 shipped {rec['L31_shipped']:.4f}  ±31 refit {rec['L31_refit']:.4f}"
                  f"  ({rec['delta_vs_L63']:+.4f}, best epoch {x['bestEpoch']})", flush=True)
    res["complete"] = len(res["spikes"]) == 7
    res["ship"] = res["complete"] and all(v["delta_vs_L63"] >= -RULE for v in res["spikes"].values())
    out.write_text(json.dumps(res, indent=1) + "\n")
    print("ship ±31:", res["ship"])


if __name__ == "__main__":
    main()
