"""Reference answers for the embed runtime: adapter-library's NumPy LoraRef on two sample emails.

    python3 -I run/embed_fixture.py

Scores the first two everyday-set email samples with the shipped emails spike, through the same prompt
layout the training code uses (browser-decider Enc.build), with the spike's temperatures, and writes
out/embed_fixture.json: {states, probs: [{question: {option: p}}]}. run/check.mjs embed compares the
ready-made page's answers with these.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

AL = experiment("adapter-library")
sys.path.insert(0, str(AL / "run"))
from pack_system import LoraRef  # noqa: E402
from shared import BD  # noqa: E402
sys.path.insert(0, str(BD / "run"))
from common import options_of  # noqa: E402
from train_enc import Enc  # noqa: E402


def main():
    web = AL / "web" / "models"
    sm = json.loads((web / "everyday" / "spikes" / "emails.json").read_text())
    b = web / sm.get("base", "base")   # the base the spike was trained on
    man = json.loads((b / "manifest.json").read_text())
    ref = LoraRef(man, (b / "weights.bin").read_bytes())
    ref.use(sm, (web / "everyday" / "spikes" / "emails.bin").read_bytes())
    tok = Enc(web / "base" / "tokenizer.json")
    samples = [c for c in json.loads((web / "everyday" / "samples.json").read_text())["cases"] if c["workflow"] == "emails"][:2]
    out = {"states": [c["state"] for c in samples], "probs": []}
    for c in samples:
        ans = {}
        for name, q in sm["questions"].items():
            opts = list(options_of(q).items())
            ids, pos = tok.build({"question": f"{q['type']}: {q['instructions']}", "options": opts, "state": c["state"]})
            start = ids.index(tok.sep, pos[-1]) + 1
            z, _ = ref.run(ids, pos, (start, len(ids) - 1))
            T = sm["temps"].get(q["type"], 1.0)
            p = np.exp(z / T - (z / T).max()); p /= p.sum()
            ans[name] = {o: float(v) for (o, _), v in zip(opts, p)}
        out["probs"].append(ans)
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "embed_fixture.json").write_text(json.dumps(out))
    print(json.dumps(out["probs"], indent=1)[:600])


if __name__ == "__main__":
    main()
