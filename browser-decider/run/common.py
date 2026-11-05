"""Data loading, prompt layout and the dataset's own scorer for browser-decider.

One example = one typed question about one case. The layout is shared by the
trainer, the ONNX export and the browser page, so it lives here only.
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "data" / "td"
WORKFLOWS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]

NOUL_DEFAULT = {
    "true": "The statement is true for this case.",
    "false": "The statement is not true for this case.",
}


def options_of(q: dict) -> dict:
    """Option -> description. Gold for noul uses true/false keys."""
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        if not crit:
            return dict(NOUL_DEFAULT)
        # noul criteria are keyed true/false (checked in load_cases)
        return {k: crit[k] for k in ("true", "false")}
    if isinstance(crit, list):  # score rubrics: level i is option "i"
        return {str(i): d for i, d in enumerate(crit)}
    return dict(crit)


def question_text(q: dict) -> str:
    return f"{q['type']}: {q['instructions']}"


def load_cases(workflow: str, split: str) -> list[dict]:
    df = pd.read_parquet(TD / workflow / f"{split}.parquet")
    out = []
    for r in df.itertuples():
        qs = json.loads(r.questions)
        gold = json.loads(r.gold)
        for name, q in qs.items():
            if q["type"] == "noul" and q.get("criteria"):
                assert set(q["criteria"]) == {"true", "false"}, (workflow, name, q["criteria"].keys())
        out.append({"id": r.id, "workflow": workflow, "state": r.state, "questions": qs, "gold": gold})
    return out


def examples(cases: list[dict]) -> list[dict]:
    ex = []
    for c in cases:
        for name, q in c["questions"].items():
            opts = options_of(q)
            g = c["gold"][name]["probabilities"]
            assert set(g) == set(opts), (c["id"], name, g.keys(), opts.keys())
            ex.append({
                "case": c["id"], "workflow": c["workflow"], "qname": name, "type": q["type"],
                "question": question_text(q), "instructions": q["instructions"], "options": list(opts.items()), "state": c["state"],
                "gold": [float(g[o]) for o in opts],
            })
    return ex


# --- the dataset's scorer, verbatim logic from data/td/score_with_jobs.py ---
def score_one(pred, gold):
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    p = [float(pred["probabilities"].get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    am_p = p.index(max(p))
    am_g = g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return am_p == am_g, kl, brier, max(p)


def summarize(rows: list[tuple]) -> dict:
    """rows: (ok, kl, brier, conf) per decision."""
    n = len(rows)
    nb = 10
    hits, tot, conf = [0] * nb, [0] * nb, [0.0] * nb
    for ok, _, _, c in rows:
        b = min(int(c * nb), nb - 1)
        tot[b] += 1
        hits[b] += int(ok)
        conf[b] += c
    ece = sum(tot[b] / n * abs(hits[b] / tot[b] - conf[b] / tot[b]) for b in range(nb) if tot[b])
    return {
        "n_decisions": n,
        "accuracy": sum(r[0] for r in rows) / n,
        "kl_from_gold": sum(r[1] for r in rows) / n,
        "brier": sum(r[2] for r in rows) / n,
        "ece": ece,
    }


def score_examples(exs: list[dict], probs: list[list[float]]) -> tuple[dict, list[tuple]]:
    rows = []
    for e, p in zip(exs, probs):
        names = [o for o, _ in e["options"]]
        rows.append(score_one({"probabilities": dict(zip(names, p))},
                              {"probabilities": dict(zip(names, e["gold"]))}))
    return summarize(rows), rows
