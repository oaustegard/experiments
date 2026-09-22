#!/usr/bin/env python3
"""Check a RESULTS.md against the protocol in CLAUDE.md: the answer first, then the
findings, then the method, then the log. Exit 1 with reasons on any failure.

  python3 _lib/results_lint.py <experiment>/RESULTS.md [...]
  python3 _lib/results_lint.py --all        # every RESULTS.md in the repo
"""
import re, sys
from pathlib import Path

ANSWER_MAX_WORDS = 250
REQUIRED = ["Answer", "Findings", "Method", "Log"]


def check(path: Path) -> list[str]:
    text = path.read_text()
    h2 = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^## (.+)$", text, re.M)]
    names = [n for _, n in h2]
    errs = []
    for i, want in enumerate(REQUIRED):
        if i >= len(names) or names[i].split(":")[0].split(" (")[0] != want:
            errs.append(f"H2 #{i+1} must be '{want}', found {names[i] if i < len(names) else 'nothing'}")
    if len(h2) >= 2 and names[0].startswith("Answer"):
        answer = text[h2[0][0]:h2[1][0]]
        words = len(re.findall(r"\b\w+\b", answer))
        if words > ANSWER_MAX_WORDS:
            errs.append(f"Answer is {words} words; limit {ANSWER_MAX_WORDS}")
        if not re.search(r"\d", answer):
            errs.append("Answer carries no number")
    if "Findings" in names:
        fi = names.index("Findings"); block = text[h2[fi][0]:h2[fi + 1][0] if fi + 1 < len(h2) else len(text)]
        if not re.search(r"^\s*1\. ", block, re.M):
            errs.append("Findings is not a numbered list")
        if not re.search(r"\(Round|\(ERRORS|\(`", block):
            errs.append("Findings cite no round or artefact")
    if "Log" in names and names.index("Log") != len(names) - 1:
        errs.append("Log must be the last H2; rounds go under it as H3")
    if re.search(r"^## Round", text, re.M):
        errs.append("a round is an H2; demote it under Log")
    return errs


def main(argv):
    paths = sorted(Path(".").glob("*/RESULTS.md")) if argv == ["--all"] else [Path(a) for a in argv]
    bad = 0
    for p in paths:
        errs = check(p)
        if errs:
            bad += 1; print(f"FAIL {p}"); [print(f"  - {e}") for e in errs]
        elif argv != ["--all"]:
            print(f"ok   {p}")
    if argv == ["--all"]:
        print(f"{len(paths) - bad}/{len(paths)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
