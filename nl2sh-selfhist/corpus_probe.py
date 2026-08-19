#!/usr/bin/env python3
"""Where does a real evaluation corpus for a shell helper come from?

Three passes of this work ran on NL2Bash, whose **60.3% `find` skew** distorted
two separate measurements: a query-independent frequency prior beat the
retrieval headline, and a constant "always answer `find`" scored 0.675 on the
gate. Two replacements were proposed and both are measured here.

**This session's own shell history** — 289 Bash calls. Flatter than NL2Bash (22
distinct leading utilities, commonest at 24.2%) but unusable as a benchmark:
only 66 are single-line and self-contained, only 26 avoid invoking this
project's own scripts, and roughly 15 of those 26 are variants of "print lines
X to Y of file F". An agent's history is file-slicing and script invocation, not
general utility composition. The Bash tool's `description` field, which would
have supplied real paired natural language, is not persisted in the transcript.

**A public corpus of real human shell sessions** — the Zenodo/UCI dataset of
hands-on cybersecurity training (record 8136017, CC-BY-4.0): 16,065 bash
commands from 275 participants, with timestamps and working directories. This
is the better instrument for the skew problem by a wide margin, and it is what
the coverage numbers below use. Its own bias is domain rather than shape:
`nmap`, `fcrackzip`, `john` and `msfconsole` are over-represented because the
participants were doing security exercises.

    python3 corpus_probe.py --cyber <dir of *-useractions.json> \\
                            --chunks ../nl2sh-retrieval/data/chunks.jsonl \\
                            --tldr <tldr pages dir>
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT = re.compile(r"\|\||&&|\||;")
ALIAS = re.compile(r"alias of `([^`]+)`")
WRAPPERS = {"sudo", "time", "nohup", "command"}


def utility(cmd: str) -> str:
    m = re.sub(r'"[^"]*"|\'[^\']*\'', '""', cmd)
    seg = SPLIT.split(m)[0]
    for tok in seg.split():
        if tok in WRAPPERS:
            continue
        if "=" in tok and not tok.startswith("-"):
            continue
        return tok.strip("()`$")
    return ""


def load_cyber(d: Path) -> list[str]:
    out = []
    for f in glob.glob(str(d / "**" / "*.json"), recursive=True):
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("cmd_type") == "bash-command" and r.get("cmd"):
                out.append(r["cmd"])
    return out


def tldr_aliases(pages: Path) -> dict[str, str]:
    """tldr ships `foo` as a stub saying it is an alias of `bar`. The corpus
    builder dropped those as stubs; resolving them instead is free coverage."""
    out = {}
    for f in glob.glob(str(pages / "**" / "*.md"), recursive=True):
        m = ALIAS.search(open(f, errors="replace").read())
        if m:
            out[Path(f).stem] = m.group(1).split()[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cyber", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results_corpus.json")
    a = ap.parse_args()

    cmds = load_cyber(a.cyber)
    freq = collections.Counter(u for u in (utility(c) for c in cmds) if u)
    total = sum(freq.values())

    documented = {json.loads(l)["utility"] for l in open(a.chunks)}
    alias = {k: v for k, v in tldr_aliases(a.tldr).items()
             if v in documented and k not in documented}

    def cov(pool):
        inc = [u for u in pool if u in documented]
        inc_a = [u for u in pool if u in documented or u in alias]
        w = sum(freq[u] for u in pool) or 1
        return {"n": len(pool),
                "utilities": round(len(inc) / len(pool), 3),
                "utilities_with_alias_fix": round(len(inc_a) / len(pool), 3),
                "by_invocation": round(sum(freq[u] for u in inc) / w, 3),
                "by_invocation_with_alias_fix": round(sum(freq[u] for u in inc_a) / w, 3)}

    tail = [u for u, c in freq.items() if c == 1]
    out = {
        "cyber_corpus": {
            "source": "Zenodo 8136017 (UCI 869), CC-BY-4.0",
            "commands": len(cmds), "unique_commands": len(set(cmds)),
            "distinct_utilities": len(freq),
            "constant_prior_commonest_utility": round(freq.most_common(1)[0][1] / total, 3),
            "commonest": freq.most_common(1)[0][0],
            "utilities_appearing_once": len(tail),
            "top_n_coverage": {str(k): round(sum(v for _, v in freq.most_common(k)) / total, 3)
                               for k in (5, 10, 20, 50, 100)},
        },
        "nl2bash_comparison": {"constant_prior": 0.603, "commonest": "find",
                               "distinct_utilities": 389},
        "documentation_coverage": {
            "doc_corpus_utilities": len(documented),
            "tldr_alias_pages_resolvable": len(alias),
            "all": cov(list(freq)),
            "top_50_by_usage": cov([u for u, _ in freq.most_common(50)]),
            "tail_used_once": cov(tail),
        },
        "undocumented_by_usage": [[u, freq[u]] for u, _ in freq.most_common()
                                  if u not in documented and u not in alias][:20],
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    c = out["cyber_corpus"]
    print(f"{c['commands']} commands, {c['distinct_utilities']} utilities, "
          f"constant prior {c['constant_prior_commonest_utility']} ({c['commonest']}) "
          f"vs NL2Bash 0.603 (find)")
    for k, v in out["documentation_coverage"].items():
        if isinstance(v, dict):
            print(f"  {k:<18} utilities {v['utilities']:.1%} "
                  f"(+alias {v['utilities_with_alias_fix']:.1%})   "
                  f"invocations {v['by_invocation']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
