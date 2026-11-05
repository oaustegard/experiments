"""Reference token ids for the page's prompt builder, from the code the spikes were trained with.

    python3 -I run/make_fixtures.py

Takes the first 4 demo cases, builds each question's sequence with browser-decider's Enc.build over
common.examples (the training path of every adapter-library spike), and writes web/fixtures.json:
{cases: [{case, q, ids, pos}]}. check.html compares the page's creator.js build() with these ids.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

BD = experiment("browser-decider")
sys.path.insert(0, str(BD / "run"))
from common import examples  # noqa: E402
from train_enc import Enc  # noqa: E402


def main():
    demo = json.loads((ROOT / "web" / "demo" / "invoice_processing.json").read_text())
    qs = demo["decision"]["questions"]
    cases = [{"id": c["id"], "workflow": demo["decision"]["id"], "state": c["state"], "questions": qs,
              "gold": {n: {"probabilities": g} for n, g in c["gold"].items()}} for c in demo["cases"][:4]]
    enc = Enc(experiment("adapter-library") / "web" / "models" / "base" / "tokenizer.json")
    out = []
    for e in examples(cases):
        ids, pos = enc.build(e)
        out.append({"case": e["case"], "q": e["qname"], "ids": ids, "pos": pos})
    (ROOT / "web" / "fixtures.json").write_text(json.dumps({"source": "browser-decider Enc.build over common.examples", "cases": out}))
    print(len(out), "sequences;", "lengths", [len(o["ids"]) for o in out][:10])


if __name__ == "__main__":
    main()
