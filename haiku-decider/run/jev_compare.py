"""Jev against the cue prompt on the same labelled answers, plus cross-validated calibration.

    python3 run/jev_compare.py      # -> results/jev_same_items.json

Jev's distributions come from dettin-68m: data/paper_jev.jsonl (the 150 whole articles) and
data/jev_sections.jsonl.gz (every section, the 395 here among them). The Haiku arms are the test-half
`v030stack` run (decide.py --stack with the cue prompt, low effort): `one` is each request's first
sample, `stack` the uncalibrated pool. `old one` is the cue-less plain single-call run on the same items.
Bundled has no per-item Jev rows; its Jev figure (0.894 on all 104 items, 2026-10-09) is the skill's
method.md number, so that row compares totals, not items.

Calibration: per-type temperatures fitted on one half of the items (split by an md5 of the id) and
applied to the other, both ways, so no answer is scored by a temperature fitted on itself.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
sys.argv = sys.argv[:1]
import prompts as P  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent / "claude-skills" / "deciding-with-confidence" / "scripts"))
import decide as D  # noqa: E402

DETTIN = HERE.parent / "dettin-68m" / "data"


def jev():
    papers = {}
    for line in open(DETTIN / "paper_jev.jsonl"):
        r = json.loads(line)
        papers[r["id"]] = {k.replace("paper_", ""): v for k, v in r["probs"].items()}
    sections = {}
    with gzip.open(DETTIN / "jev_sections.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            sections[r["id"]] = r["probs"]
    return {"papers": papers, "sections": sections}


def strip(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "right"}


def cv_ece(rows):
    """rows: [(item_id, type, probs, label)] -> ECE after 2-fold per-type temperatures."""
    fold = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 2  # noqa: E731
    out = []
    for t in {r[1] for r in rows}:
        sub = [r for r in rows if r[1] == t]
        for j in (0, 1):
            train = [(p, y) for i, _, p, y in sub if fold(i) != j]
            T = D.fit_temperature(train) if len(train) >= 20 else 1.0
            out += [(D.temper(D.floored(p), T), y) for i, _, p, y in sub if fold(i) == j]
    return P.metrics(out)


def main():
    J = jev()
    res = {}
    for s in ("papers", "sections"):
        new, old = P.load(s, "v030stack", "test"), P.load(s, "plain", "test")
        keys = [k for k in new if k in old and k[0] in J[s] and k[1] in J[s][k[0]]]
        lab = {k: new[k]["label"] for k in keys}
        arms = {"jev": lambda k: J[s][k[0]][k[1]], "old one": lambda k: old[k]["raw"],
                "one": lambda k: new[k]["per_sample"][0], "stack": lambda k: new[k]["raw"]}
        res[s] = {"answers": len(keys), "items": len({k[0] for k in keys})}
        for name, f in arms.items():
            m = P.metrics([(f(k), lab[k]) for k in keys])
            res[s][name] = strip(m)
            if name in ("one", "stack"):
                res[s][name]["cv_calibrated"] = strip(cv_ece([(k[0], new[k]["type"], f(k), lab[k]) for k in keys]))
        lo, hi = P.boot_diff(P.metrics([(arms["stack"](k), lab[k]) for k in keys])["right"],
                             P.metrics([(arms["jev"](k), lab[k]) for k in keys])["right"])
        res[s]["stack_minus_jev_ci90"] = [round(lo, 4), round(hi, 4)]
        lo, hi = P.boot_diff(P.metrics([(arms["one"](k), lab[k]) for k in keys])["right"],
                             P.metrics([(arms["jev"](k), lab[k]) for k in keys])["right"])
        res[s]["one_minus_jev_ci90"] = [round(lo, 4), round(hi, 4)]
    new = P.load("bundled", "v030stack", "all")
    if new:
        keys = sorted(new)
        lab = {k: new[k]["label"] for k in keys}
        res["bundled"] = {"answers": len(keys), "jev_reported": 0.894,
                          "one": strip(P.metrics([(new[k]["per_sample"][0], lab[k]) for k in keys])),
                          "stack": strip(P.metrics([(new[k]["raw"], lab[k]) for k in keys]))}
        for t in ("choice", "predicate", "score"):
            kk = [k for k in keys if new[k]["type"] == t]
            res["bundled"][f"stack_{t}"] = strip(P.metrics([(new[k]["raw"], lab[k]) for k in kk]))
            res["bundled"][f"one_{t}"] = strip(P.metrics([(new[k]["per_sample"][0], lab[k]) for k in kk]))
    (HERE / "results" / "jev_same_items.json").write_text(json.dumps(res, indent=1) + "\n")
    for s, v in res.items():
        print(s, json.dumps({k: (x["acc"] if isinstance(x, dict) and "acc" in x else x) for k, x in v.items()}))


if __name__ == "__main__":
    main()
