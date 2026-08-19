#!/usr/bin/env python3
"""Does agreement between two independently-derived routers gate like agreement
between two models does?

`monad-bsky/synergy.py` found that two small models naming the same tool were
right 0.880 of the time at 0.455 coverage, beating a calibrated confidence head
at matched coverage — but it cost running both models, 11x latency. Both routers
here are deterministic and cost microseconds, so if the same effect holds the
gate is nearly free.

The two arms are not independent in the way two models are: they share this
repo's cue layer and the catalogue. They differ in everything above that — one
is searched under a covering objective on family A, the other is written by
hand from the schemas.

    python3 agreement.py
"""

from __future__ import annotations

import json
from pathlib import Path

from handwritten import HandRouter
from router import Router

HERE = Path(__file__).resolve().parent

SPLITS = {
    "family B (held-out)": HERE / "data" / "family_b.jsonl",
    "wild (hand-authored)": HERE / "wild.jsonl",
}


def main() -> int:
    fitted = Router(HERE / "rules_schema.json")
    hand = HandRouter()
    out = {}
    hdr = f"{'split':<22}{'gate':<26}{'coverage':>10}{'precision':>11}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for sname, path in SPLITS.items():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        on = [r for r in rows if r.get("label")]
        pairs = [(r["label"], fitted.route(r["query"]), hand.route(r["query"])) for r in on]
        n = len(on)

        def report(tag, sel):
            hit = [g == a for g, a, _ in sel] if tag.startswith("fitted") else \
                  [g == b for g, _, b in sel]
            cov, prec = len(sel) / n, (sum(hit) / len(sel) if sel else 0.0)
            print(f"{sname:<22}{tag:<26}{cov:>10.3f}{prec:>11.3f}{len(sel):>6}")
            out.setdefault(sname, {})[tag] = {"coverage": round(cov, 4),
                                              "precision": round(prec, 4), "n": len(sel)}

        report("fitted, no gate", [p for p in pairs if p[1] is not None])
        report("hand, no gate", [p for p in pairs if p[2] is not None])
        agree = [p for p in pairs if p[1] is not None and p[1] == p[2]]
        cov = len(agree) / n
        prec = sum(g == a for g, a, _ in agree) / len(agree) if agree else 0.0
        print(f"{sname:<22}{'both agree':<26}{cov:>10.3f}{prec:>11.3f}{len(agree):>6}")
        out.setdefault(sname, {})["both agree"] = {"coverage": round(cov, 4),
                                                   "precision": round(prec, 4),
                                                   "n": len(agree)}
        # Where they disagree, is either one worth taking?
        dis = [p for p in pairs if p[1] is not None and p[2] is not None and p[1] != p[2]]
        if dis:
            pf = sum(g == a for g, a, _ in dis) / len(dis)
            ph = sum(g == b for g, _, b in dis) / len(dis)
            print(f"{sname:<22}{'disagree: fitted right':<26}{len(dis) / n:>10.3f}{pf:>11.3f}{len(dis):>6}")
            print(f"{sname:<22}{'disagree: hand right':<26}{len(dis) / n:>10.3f}{ph:>11.3f}{len(dis):>6}")
            out[sname]["disagree"] = {"n": len(dis), "fitted_right": round(pf, 4),
                                      "hand_right": round(ph, 4)}
        print()
    (HERE / "results_agreement.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
