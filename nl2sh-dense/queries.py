#!/usr/bin/env python3
"""The query distributions the abstention threshold has to hold across.

Issue #47's live run exposed that the gate's **absolute** margin (top1-top2 over
per-utility BM25 scores) does not transfer: it was calibrated at >=5 on the
security corpus where scores run 11-43, and on everyday requests where they run
0.2-2.8 *everything* abstains. A threshold is only useful if one operating point
works on more than the distribution it was fitted to, so calibration needs at
least two distributions, and they need to differ in phrasing style as well as in
subject.

Three are available without authoring a new eval:

* **cyber** — `nl2sh-selfhist/cyber_nl.json`. Real commands from the Zenodo/UCI
  hands-on-security corpus, natural language written by Gemini and instructed not
  to name the utility. The primary eval; n=34 after dropping the 4 rows that
  named the utility anyway. Constant-utility prior is low (the corpus's own is
  0.189).
* **selfhist** — the 14 hand-authored pairs in `nl2sh-selfhist/selfhist_eval.py`,
  written in the task shapes an agent's real shell history showed (file slicing,
  grep, ls). This is the everyday distribution whose scores collapsed in the demo.
  Hand-authored, so a probe rather than a benchmark.
* **nl2bash** — a sample of NL2Bash. Human-curated but annotator-written *from
  the command*, and 60.3% `find`, which is exactly the skew this line of work
  moved away from. It is here for phrasing variety and sample size only, always
  reported beside its constant-utility prior, and never as a headline.

Every set is returned as `[{nl, utility, names_utility}]`, matching
`cyber_nl.json`'s shape, so one calibration loop reads all three.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
SELFHIST = experiment("nl2sh-selfhist")
sys.path.insert(0, str(RETRIEVAL))
import pleias_gate as G  # noqa: E402


def _names_utility(nl: str, utility: str) -> bool:
    """Same test `gen_nl.py` applies: does the request contain the answer?"""
    return re.search(rf"\b{re.escape(utility)}\b", nl, re.I) is not None


#: The original 38-row eval plus the 149-row extension sampled by
#: `sample_cyber.py` under the same tiered protocol. Both are read by default;
#: pass `paths` to use one alone.
CYBER_NL = (SELFHIST / "cyber_nl.json", HERE / "cyber_nl_ext.json")


def cyber(tldr: dict | None = None, paths=CYBER_NL) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        if not Path(path).exists():
            continue
        for r in json.loads(Path(path).read_text()):
            if not r.get("nl") or r["nl"] in seen:
                continue
            if tldr is not None and r["utility"] not in tldr:
                continue
            seen.add(r["nl"])
            rows.append(r)
    return rows


def selfhist(tldr: dict | None = None) -> list[dict]:
    sys.path.insert(0, str(SELFHIST))
    import selfhist_eval as S

    out = []
    for nl, cmd in S.PAIRS:
        u = G.gold_utility(cmd)
        if tldr is not None and u not in tldr:
            continue
        out.append({"nl": nl, "utility": u, "names_utility": _names_utility(nl, u)})
    return out


def nl2bash(root: Path, n: int = 300, seed: int = 20260820,
            tldr: dict | None = None, drop_find: bool = False) -> list[dict]:
    nls = (root / "all.nl").read_text(errors="replace").splitlines()
    cms = (root / "all.cm").read_text(errors="replace").splitlines()
    pool = []
    for nl, cm in zip(nls, cms):
        u = G.gold_utility(cm)
        if not u or (tldr is not None and u not in tldr):
            continue
        if drop_find and u == "find":
            continue
        pool.append({"nl": nl, "utility": u, "names_utility": _names_utility(nl, u)})
    random.Random(seed).shuffle(pool)
    return pool[:n]


def constant_prior(rows: list[dict]) -> tuple[str, float]:
    """The commonest gold utility and its share — the score to beat by doing nothing."""
    from collections import Counter

    c = Counter(r["utility"] for r in rows)
    if not c:
        return ("", 0.0)
    u, k = c.most_common(1)[0]
    return (u, round(k / len(rows), 3))


def load_all(tldr: dict, nl2bash_root: Path | None = None,
             nl2bash_n: int = 300) -> dict[str, list[dict]]:
    sets = {"cyber": cyber(tldr), "selfhist": selfhist(tldr)}
    if nl2bash_root and nl2bash_root.exists():
        sets["nl2bash"] = nl2bash(nl2bash_root, nl2bash_n, tldr=tldr)
    return sets
