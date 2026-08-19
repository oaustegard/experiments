#!/usr/bin/env python3
"""Extract literal parameter values that are already present in a shell request.

WHY THIS EXISTS
---------------
`monad-bsky` measured a 56M model copying identifiers correctly only ~51% of the
time, and that is the one finding that transferred unconditionally: a model must
never retype an identifier it was given. In a shell helper that stops being a
quality property and becomes a safety property -- a model that "helpfully"
regenerates `/etc/nginx.conf` as `/etc/nginx/nginx.conf`, or `rm`s `*.log`
instead of `*.log.1`, is worse than no helper. So every literal that the user
already typed should be lifted out of the request by deterministic code, bound
to a character span, and handed to the generator as a fixed slot rather than as
something to be reproduced from memory.

This module is that lifting step. It is deliberately conservative: it fires only
on *structural* evidence (a quote, a slash, a dot-extension, a `$`, a digit plus
a unit, a cue word plus a number). It does not try to guess that `repogroup` in
"changes group ownership to repogroup" is a group name unless the request
punctuates it. That conservatism is the point -- a parameter extractor that
guesses has given up the property it exists to provide -- but it also bounds
recall, and `eval_extract.py` measures exactly where that bound lands.

Design notes
------------
* Every match carries `(start, end)` into the original string. Downstream code
  is expected to slice the request, never to re-type the value.
* Candidates from all patterns are pooled and resolved by a single greedy
  longest-span-wins pass, with ties broken by kind specificity. This is how a
  quoted `"path/to/file.gz"` comes back as a `path` rather than as a generic
  `literal`, while a quoted `"%Y-%m-%d %H:%M:%S"` stays one `literal` instead of
  fragmenting into a `date` and two `time`s.
* Patterns that would fire on ordinary English are gated, not dropped:
  `and/or` is rejected as a path because both segments are bare alphabetic and
  there is no leading `/`; `e.g` is rejected as a filename because stem and
  extension are both one character.

Usage
-----
    python3 extract_params.py --text "Find all *.mov files under /mnt/raid"
    python3 extract_params.py --file requests.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Dict, List, NamedTuple


class Span(NamedTuple):
    """One extracted value and where it came from in the request."""

    kind: str
    value: str
    start: int
    end: int


# Lower number == more specific. Used only to break span-length ties, so that a
# quoted region whose contents are exactly a path is reported as a path.
KIND_PRIORITY = {
    "url": 0,
    "ip": 0,
    "var": 0,
    "path": 1,
    "glob": 1,
    "filename": 2,
    "extension": 2,
    "hostname": 2,
    "git_ref": 2,
    "signal": 2,
    "port": 3,
    "size": 3,
    "duration": 3,
    "date": 3,
    "time": 3,
    "pid": 3,
    "perm": 3,
    "identifier": 4,
    "user": 4,
    "process": 4,
    "number": 5,
    "literal": 9,  # generic quoted span; loses every tie
}

# --------------------------------------------------------------------------
# Quoting
# --------------------------------------------------------------------------
# The lookarounds are what keep "don't" and "file's" from opening a quote.
_QUOTED = [
    re.compile(r"(?<![A-Za-z0-9])'([^'\n]{1,160})'(?![A-Za-z0-9])"),
    re.compile(r'"([^"\n]{1,160})"'),
    re.compile(r"\u201c([^\u201d\n]{1,160})\u201d"),
]
# NL2Bash uses ` as an apostrophe ("don`t"), so backtick pairs are only trusted
# when the enclosed text has no whitespace.
_BACKTICKED = re.compile(r"`([^`\s]{1,80})`")

# --------------------------------------------------------------------------
# Value patterns.  Each entry: (kind, compiled regex, capture group).
# --------------------------------------------------------------------------
_PATHCHAR = r"[\w.\-*?~$@%+=\[\]{}!]"

PATTERNS: List[tuple] = [
    ("url", re.compile(
        r"\b(?:https?|ftps?|sftp|ssh|git|rsync|file|smb|nfs|mailto):"
        r"//?[^\s'\"<>()\[\],]+", re.I), 0),
    ("ip", re.compile(
        r"(?<![\w.])((?:\d{1,3}\.){3}\d{1,3})(?:/(\d{1,2}))?(?![\w.])"), 0),
    ("ip", re.compile(r"(?<![\w:])((?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4})(?![\w:])"), 1),
    ("var", re.compile(r"\$\{[A-Za-z_]\w*(?:[^{}\n]{0,40})?\}|\$[A-Za-z_]\w*|\$\d+|\$[@*#?!$]"), 0),
    ("path", re.compile(rf"(?<![\w:/]) ( (?: {_PATHCHAR}+ )? (?: / {_PATHCHAR}+ )+ /? )", re.X), 1),
    ("glob", re.compile(r"(?<![\w/*])((?:[\w.\-+]*\*)+[\w.\-+]*)(?![\w])"), 1),
    ("hostname", re.compile(
        r"(?<![\w./@-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
        r"(?:com|org|net|edu|gov|mil|int|io|co|uk|de|fr|jp|ru|cn|br|in|au|ca|nl|se|no|fi|"
        r"dk|ch|be|at|pl|cz|es|it|za|tv|cc|ly|sh|gl|me|us|info|biz|name|local|dev|app|xyz))"
        r"(?![\w.-])", re.I), 1),
    ("hostname", re.compile(r"\blocalhost\b", re.I), 0),
    ("filename", re.compile(r"(?<![\w./*])([\w\-+]{1,60}\.[A-Za-z0-9]{1,8})(?![\w.])"), 1),
    ("extension", re.compile(r"(?<![\w*./])(\.[A-Za-z][A-Za-z0-9]{0,11})(?![\w.])"), 1),
    ("port", re.compile(r"\bports?\s+(?:number\s+)?(\d{1,5})\b", re.I), 1),
    ("size", re.compile(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s?([KkMmGgTtPp][Ii]?[Bb]|[KMGTP]B?|bytes?|blocks?|kilobytes?|"
        r"megabytes?|gigabytes?|terabytes?)(?![\w])"), 1),
    ("duration", re.compile(
        r"(?<![\w.])(\d+)\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
        re.I), 1),
    ("date", re.compile(r"(?<![\w-])(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})(?![\w-])"), 1),
    ("time", re.compile(r"(?<![\w:])(\d{1,2}:\d{2}(?::\d{2})?)(?![\w:])"), 1),
    ("pid", re.compile(r"\b(?:pid|process\s+id|process\s+ID)\s*#?\s*(\d+)\b", re.I), 1),
    ("pid", re.compile(r"\bprocess\s+(\d{2,})\b"), 1),
    ("signal", re.compile(r"\bSIG([A-Z]{2,10})\b"), 1),
    ("git_ref", re.compile(r"\bHEAD(?:[~^]\d*)*"), 0),
    ("git_ref", re.compile(r"\b(?:origin|upstream)/[\w./-]+"), 0),
    ("git_ref", re.compile(r"\brefs/[\w./-]+"), 0),
    ("git_ref", re.compile(
        r"\b(?:branch|tag|commit|revision)\s+(?:named\s+|called\s+)?[\"'`]?([\w][\w./-]{1,60})[\"'`]?",
        re.I), 1),
    ("git_ref", re.compile(r"(?<![\w.])((?=[0-9a-f]{7,40}\b)(?=[a-f]*\d)(?=\d*[a-f])[0-9a-f]{7,40})(?![\w.])"), 1),
    ("perm", re.compile(
        r"\b(?:chmod|permissions?|mode|umask|octal)\b[^.\n]{0,20}?\b(0?[0-7]{3,4})\b", re.I), 1),
    ("perm", re.compile(r"(?<![\w.])(0?[0-7]{3,4})\s+(?:permissions?|mode|octal)\b", re.I), 1),
    ("perm", re.compile(r"(?<![\w])([ugoa]*[+\-=][rwxstXugo]{1,6})(?![\w])"), 1),
    ("user", re.compile(
        r"\b(?:user|username|user\s+name|owner|group|owned\s+by|belonging\s+to)\s+"
        r"(?:named\s+|called\s+|user\s+)?[\"'`]?([A-Za-z_][\w.\-]{0,31})[\"'`]?", re.I), 1),
    ("process", re.compile(
        r"\bprocess(?:es)?\s+(?:named|called|matching|whose\s+(?:command|name)"
        r"(?:\s+line)?\s+(?:contains|includes|is))\s+[\"'`]?([\w.\-/]+)[\"'`]?", re.I), 1),
    ("process", re.compile(
        r"\b(?:kill|killall|terminate|pkill)\s+(?:all\s+)?(?:the\s+)?[\"'`]?([\w.\-]+)[\"'`]?\s+process",
        re.I), 1),
    ("identifier", re.compile(r"(?<![\w./-])([A-Za-z][\w]*(?:[_\-][A-Za-z0-9]+)+)(?![\w./-])"), 1),
    ("identifier", re.compile(r"(?<![\w./])((?=[A-Za-z0-9]{4,24}\b)(?=[^\s]*\d)[a-z]+\d[A-Za-z0-9]*)(?![\w./])"), 1),
    ("number", re.compile(
        r"\b(?:first|last|top|maximum|max|minimum|min|depth|level|levels|limit|line|lines|"
        r"column|columns|character|characters|byte|bytes|field|fields|time|times|count|"
        r"most|every)\s+(\d+)\b", re.I), 1),
    ("number", re.compile(
        r"(?<![\w.])(\d+)\s+(?:lines?|levels?|characters?|chars?|bytes?|files?|times?|"
        r"columns?|fields?|processes|directories|entries|results?)\b", re.I), 1),
    ("number", re.compile(r"(?<![\w.$/-])(\d+(?:\.\d+)?)(?![\w.$/-])"), 1),
]

# Words that follow "user"/"group"/"owner" in prose rather than naming one.
_USER_STOP = {
    "processes", "process", "input", "id", "ids", "name", "names", "account",
    "accounts", "space", "defined", "agent", "directory", "directories", "of",
    "and", "or", "the", "a", "an", "is", "are", "who", "whose", "which", "that",
    "ownership", "owner", "group", "groups", "user", "users", "in", "to", "for",
    "by", "with", "from", "on", "as", "it", "its", "all", "each", "no", "not",
    "file", "files", "home", "level", "list", "mode", "read", "write", "run",
    "owned", "belonging", "current", "specified", "given", "same", "other",
    "matching", "named", "called", "containing", "whose", "having", "only",
}
_FILENAME_STOP = {"e.g", "i.e", "etc", "vs", "a.m", "p.m", "u.s"}
_GITREF_STOP = {
    "name", "names", "the", "a", "an", "of", "in", "to", "for", "and", "or",
    "with", "from", "that", "this", "is", "are", "all", "each", "current",
    "message", "messages", "history", "log", "hash", "id",
}
_PROTO_ONLY = re.compile(r"^\w+://?$")


def _plausible_path(text: str) -> bool:
    """Reject `and/or`, `input/output`, `24/7` -- prose that owns a slash.

    A slash run is only a path if it is anchored (`/x`, `./x`, `~/x`), or has
    three or more segments, or some segment carries a non-alphabetic character.
    """
    if text[0] in "/~" or text.startswith("./") or text.startswith("../"):
        return True
    segs = [s for s in text.split("/") if s]
    if len(segs) >= 3:
        return True
    return any(not s.isalpha() for s in segs)


def _english_compound(text: str) -> bool:
    """`non-zero`, `read-only`, `sub-folder` are prose, not identifiers.

    Hyphenated all-alphabetic compounds are overwhelmingly English in request
    text; underscores, digits and dots are what actually mark an identifier.
    A quoted compound still reaches the caller as a `literal`, so this filter
    costs nothing that the request itself punctuated.
    """
    return "_" not in text and text.replace("-", "").isalpha()


def _valid_ipv4(text: str) -> bool:
    head = text.split("/")[0]
    parts = head.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


_UNITS: Dict[tuple, str] = {}


def _candidates(text: str) -> List[Span]:
    out: List[Span] = []
    _UNITS.clear()

    for rx in _QUOTED:
        for m in rx.finditer(text):
            inner = m.group(1).strip()
            if inner:
                s = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
                out.append(Span("literal", inner, s, s + len(inner)))
    for m in _BACKTICKED.finditer(text):
        out.append(Span("literal", m.group(1), m.start(1), m.end(1)))

    for kind, rx, grp in PATTERNS:
        for m in rx.finditer(text):
            if m.group(grp) is None:
                continue
            value, start, end = m.group(grp), m.start(grp), m.end(grp)
            # Trailing sentence punctuation is never part of the value.
            while value and value[-1] in ".,;:!?)":
                if kind in ("extension", "filename", "date", "time", "number",
                            "duration", "size", "ip") and value[-1] == ".":
                    break
                value, end = value[:-1], end - 1
            if not value:
                continue
            if kind == "path" and not _plausible_path(value):
                continue
            if kind == "glob" and not any(c.isalpha() for c in value):
                continue
            if kind == "identifier" and _english_compound(value):
                continue
            if kind == "ip" and "." in value and not _valid_ipv4(value):
                continue
            if kind == "ip" and ":" in value and value.count(":") < 2:
                continue
            if kind == "filename" and value.lower() in _FILENAME_STOP:
                continue
            if kind == "filename":
                stem, _, ext = value.rpartition(".")
                if len(stem) <= 1 and len(ext) <= 1:
                    continue
            if kind == "user" and value.lower() in _USER_STOP:
                continue
            if kind == "git_ref" and value.lower() in _GITREF_STOP:
                continue
            if kind == "url" and _PROTO_ONLY.match(value):
                continue
            if kind == "size" and re.fullmatch(r"\d+(?:\.\d+)?\s?[Bb]", value):
                continue
            if kind == "size" and m.lastindex and m.lastindex >= 2:
                # The span is the number; the unit is a classification of it.
                # "10KB" -> value "10", unit "KB", because commands write +10k.
                _UNITS[(start, end)] = m.group(2)
            out.append(Span(kind, value, start, end))
    return out


def extract(text: str) -> Dict[str, List[dict]]:
    """Return ``{kind: [{value, start, end}, ...]}`` for one request.

    Overlapping candidates are resolved greedily: longest span first, ties to
    the more specific kind. Identical (kind, value) pairs are emitted once per
    distinct span, so a value repeated in the request keeps both spans.
    """
    cands = _candidates(text)
    cands.sort(key=lambda c: (-(c.end - c.start), KIND_PRIORITY.get(c.kind, 9), c.start))

    taken: List[Span] = []
    occupied: List[tuple] = []
    for c in cands:
        if any(c.start < e and s < c.end for s, e in occupied):
            continue
        occupied.append((c.start, c.end))
        taken.append(c)

    taken.sort(key=lambda c: c.start)
    result: Dict[str, List[dict]] = {}
    for c in taken:
        entry = {"value": c.value, "start": c.start, "end": c.end}
        if c.kind == "size" and (c.start, c.end) in _UNITS:
            entry["unit"] = _UNITS[(c.start, c.end)]
        result.setdefault(c.kind, []).append(entry)
    return result


def values(text: str) -> List[str]:
    """Flat list of extracted value strings, order of appearance."""
    return [v["value"] for spans in extract(text).values() for v in spans]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text", help="a single request to extract from")
    ap.add_argument("--file", help="file of requests, one per line")
    ap.add_argument("--json", action="store_true", help="emit JSON lines")
    args = ap.parse_args()

    if args.text:
        requests = [args.text]
    elif args.file:
        requests = [ln.rstrip("\n") for ln in open(args.file, encoding="utf-8") if ln.strip()]
    else:
        requests = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]

    for req in requests:
        got = extract(req)
        if args.json:
            print(json.dumps({"request": req, "params": got}, ensure_ascii=False))
        else:
            print(req)
            for kind, spans in sorted(got.items()):
                for sp in spans:
                    print(f"    {kind:<10} {sp['value']!r}  [{sp['start']}:{sp['end']}]")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
