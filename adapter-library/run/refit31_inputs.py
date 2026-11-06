"""Inputs for run/refit31_job.js: each shipped spike's training and validation cases, split as when it was trained.

    python3 -I run/refit31_inputs.py SET > out/base31/refit_<SET>.json     # SET: workflows | everyday

Workflows spikes (run/lora.py, QW-lora-<w>-s0): train cases sorted by id, shuffled with
Random("<w>-0"), the first 30 validate. Everyday spikes (run/everyday_train.py, E-<d>): train cases sorted
by id, shuffled with Random("everyday-<d>"), the first max(4, n // 10) validate. Gold is each question's
distribution over its options, in the creator page's case format.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_system import WEB, everyday_cases  # noqa: E402
from shared import load_cases  # noqa: E402


def case(c):
    return {"id": c["id"], "state": c["state"], "gold": {q: g["probabilities"] for q, g in c["gold"].items()}}


def main():
    set_ = sys.argv[1]
    system = json.loads((WEB / set_ / "system.json").read_text())
    out = {"set": set_, "spikes": []}
    for s in system["spikes"]:
        w = s["id"]
        if set_ == "workflows":
            cs = sorted(load_cases(w, "train"), key=lambda c: c["id"])
            random.Random(f"{w}-0").shuffle(cs)
            nval = 30
        else:
            cs = sorted(everyday_cases(w, "train"), key=lambda c: c["id"])
            random.Random(f"everyday-{w}").shuffle(cs)
            nval = max(4, len(cs) // 10)
        out["spikes"].append({"id": w, "file": s["file"], "val": [case(c) for c in cs[:nval]], "train": [case(c) for c in cs[nval:]]})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
