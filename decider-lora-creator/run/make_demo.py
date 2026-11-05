"""Write the page's demo decision: one Typed Decisions workflow as a creator dataset.

    python3 -I run/make_demo.py [WORKFLOW]      # default invoice_processing

Writes web/demo/<workflow>.json:
  {decision: {id, description, questions}, cases: [{id, state, gold: {q: {option: p}}, split}],
   init: [ids of the library spikes to merge as the warm start]}
Cases keep the dataset's train/test split. The warm start is the other three workflow spikes, as in
adapter-library's growth arm G: the page starts a new spike on a workflow the library has not seen.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

TD = experiment("browser-decider") / "data" / "td"
SYSTEM = experiment("adapter-library") / "web" / "models" / "workflows" / "system.json"


def main():
    import pandas as pd
    w = sys.argv[1] if len(sys.argv) > 1 else "invoice_processing"
    system = json.loads(SYSTEM.read_text())
    desc = next(s["description"] for s in system["spikes"] if s["id"] == w)
    cases, questions = [], None
    for split in ("train", "test"):
        df = pd.read_parquet(TD / w / f"{split}.parquet")
        for r in df.itertuples():
            qs = json.loads(r.questions)
            questions = questions or qs
            assert qs == questions, r.id
            gold = {n: {o: round(float(p), 6) for o, p in g["probabilities"].items()} for n, g in json.loads(r.gold).items()}
            cases.append({"id": r.id, "state": r.state, "gold": gold, "split": split})
    out = {"decision": {"id": w, "description": desc, "questions": questions}, "cases": cases,
           "init": [s["id"] for s in system["spikes"] if s["id"] != w],
           "source": f"LocalLLaMA/typed-decisions, workflow {w} (browser-decider/data/td)"}
    p = ROOT / "web" / "demo" / f"{w}.json"
    p.write_text(json.dumps(out, separators=(",", ":")))
    print(p, len(cases), f"{p.stat().st_size / 1e6:.2f} MB", list(questions))


if __name__ == "__main__":
    main()
