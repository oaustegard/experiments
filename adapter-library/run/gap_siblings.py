"""Post hoc (after round 4, not registered): what answers a case from a family no spike was trained on?

    python3 -I run/gap_siblings.py

For each family F, the two sibling sub-spikes of its workflow (trained on the other two families) are
scored on F's test cases: each alone, their averaged distributions, and weighted by the round 4 router's
probabilities for them (fold F, option set A). F's own sub-spike and the whole-workflow spike are
references. Writes results/gap_siblings.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import FAM_IDS, FAM_WORKFLOW, ROOT, WORKFLOWS, examples, family_map, load_cases, score_examples  # noqa: E402

OUT, RES = ROOT / "out", ROOT / "results"


def main():
    fm = family_map()
    TE = {w: examples(load_cases(w, "test")) for w in WORKFLOWS}
    P = lambda n: [np.array(p) for p in json.loads((OUT / "probs" / f"{n}.json").read_text())]
    sub = {f: P(f"R-sub-{f}") for f in FAM_IDS}
    whole = {w: P(f"S-lora-{w}-s0") for w in WORKFLOWS}
    rows = {}
    for i, F in enumerate(FAM_IDS):
        w = FAM_WORKFLOW[F]
        sibs = [f for f in FAM_IDS if FAM_WORKFLOW[f] == w and f != F]
        r = json.loads((OUT / "probs" / f"R-router-F{i:02d}.json").read_text())
        rp = {c: np.array(p) for c, p in zip(r["cases"], r["probs_A"])}
        idx = [k for k, e in enumerate(TE[w]) if fm[e["case"]] == F]
        ex = [TE[w][k] for k in idx]
        acc = lambda ps: score_examples(ex, [p.tolist() for p in ps])[0]["accuracy"]

        def routed(k):
            wt = np.array([rp[TE[w][k]["case"]][r["options_A"].index(s)] if s in r["options_A"] else 0.0 for s in sibs])
            wt = wt / wt.sum() if wt.sum() > 0 else np.full(len(sibs), 1 / len(sibs))
            return sum(a * sub[s][k] for a, s in zip(wt, sibs))

        sib_acc = [acc([sub[s][k] for k in idx]) for s in sibs]
        rows[F] = {"own_sub_spike": acc([sub[F][k] for k in idx]), "whole_workflow_spike": acc([whole[w][k] for k in idx]),
                   "best_sibling": max(sib_acc), "worst_sibling": min(sib_acc),
                   "average_of_siblings": acc([sum(sub[s][k] for s in sibs) / len(sibs) for k in idx]),
                   "router_weighted_siblings": acc([routed(k) for k in idx]), "n_cases": len(set(e["case"] for e in ex))}
        print(F, {k: round(v, 3) for k, v in rows[F].items()}, flush=True)
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values())) if k != "n_cases"}
    (RES / "gap_siblings.json").write_text(json.dumps({"mean": mean, "families": rows}, indent=1))
    print(json.dumps(mean, indent=1))


if __name__ == "__main__":
    main()
