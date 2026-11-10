#!/usr/bin/env python3
"""Check that the arm B writer brief contains none of the tell strings.

Two sources of tell strings:

1. Every regex in declaude_lint's RULES table (the lexical tells; snapshot of
   declauding 0.13.2 in context/declaude_lint.py). The lint's structural
   scanners (anaphora, echo, fragment runs, density) find shapes, not strings;
   they are reported for information and do not gate.
2. Every quoted string that muninn-voice-signature lists as a tell: quotes
   after "Scan for", "Not in use", "Zero bird idioms" or "Still banned" in a
   paragraph, and the quote right after each "rather than" (the rejected half
   of a pair). Snapshot in context/voice_signature.md.

    python3 check_brief.py [BRIEF]      # default: brief_b.md; exit 1 on any hit
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTX = HERE / "context"


def load_lint():
    spec = importlib.util.spec_from_file_location("declaude_lint", CTX / "declaude_lint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QUOTE_RE = re.compile(r"[\"“]([^\"”]{2,120})[\"”]")
TRIGGERS = ("Scan for", "Not in use", "Zero bird idioms", "Still banned")


# Quoted strings that follow a trigger but are the endorsed form, or a word the
# practice says to keep. Each was read in context/voice_signature.md.
NOT_TELLS = {
    "within a point": "header practice: what the section text says",
    "nearly match": "header practice: the endorsed header wording",
    "match": "header practice: the over-claim, but too common a word to ban alone",
    "where the macro studies diverge": "endorsed noun-phrase label for PR bodies and docs",
    "that": "brevity practice: the word to keep",
}


def voice_tells(text: str) -> list[str]:
    """Quoted tell strings from the voice signature, normalised."""
    out: list[str] = []
    # Only the rule sections; EXAMPLES holds before/after paragraphs.
    body = text.split("## EXAMPLES")[0]
    for para in re.split(r"\n\s*\n|\n- ", body):
        starts = [para.find(t) for t in TRIGGERS if t in para]
        if starts:
            for m in QUOTE_RE.finditer(para[min(starts):]):
                out.append(m.group(1))
        for m in re.finditer(r"rather than\s+[\"“]([^\"”]{2,120})[\"”]", para):
            out.append(m.group(1))
        for m in re.finditer(r"/\s+[\"“]([^\"”]{2,120})[\"”]", para):
            out.append(m.group(1))
    tells: list[str] = []
    for q in out:
        for part in q.split(" / "):
            s = part.strip().strip("…").strip().rstrip(".,;:").strip()
            s = s.replace("’", "'")
            if s.lower() in NOT_TELLS:
                continue
            if len(s) >= 3 and s.lower() not in {t.lower() for t in tells}:
                tells.append(s)
    return tells


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).lower()


def check(brief_path: Path) -> int:
    lint = load_lint()
    text = brief_path.read_text()
    hits: list[str] = []

    # 1. lint RULES, the lexical tells
    for i, line in enumerate(text.splitlines(), 1):
        for cat, rx, note in lint.COMPILED:
            for m in rx.finditer(line):
                hits.append(f"lint  line {i:3d}  [{cat}] {m.group(0)!r}  ({note})")

    # 2. voice-signature quoted tells
    tells = voice_tells((CTX / "voice_signature.md").read_text())
    flat = norm(text)
    for t in tells:
        pat = r"(?<![a-z0-9])" + re.escape(norm(t)) + r"(?![a-z0-9])"
        if re.search(pat, flat):
            hits.append(f"voice tell {t!r}")

    # Structural scanners: information only
    shapes = []
    for fn in ("_anaphora_runs", "_echo_runs", "_question_runs", "_fragment_runs"):
        f = getattr(lint, fn, None)
        if f:
            shapes += [f"{fn.strip('_')}: {h.get('match', h.get('note'))!r}" for h in f(text)]

    words = len(re.findall(r"\b\w+\b", text))
    print(f"brief: {brief_path.name}, {words} words")
    print(f"lint RULES checked: {len(lint.COMPILED)}; voice-signature tell strings checked: {len(tells)}")
    print(f"tell strings found: {len(hits)}")
    for h in hits:
        print("  " + h)
    print(f"structural shapes (information, not gated): {len(shapes)}")
    for s in shapes:
        print("  " + s)
    return 1 if hits else 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "brief_b.md"
    if "--list-tells" in sys.argv:
        for t in voice_tells((CTX / "voice_signature.md").read_text()):
            print(t)
        sys.exit(0)
    sys.exit(check(p))
