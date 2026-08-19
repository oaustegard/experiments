#!/usr/bin/env python3
"""Turn tldr pages and roff man pages into retrieval chunks for an NL2SH helper.

`nl2sh-scoping` established the shape of the problem: the difficulty is
*utility selection* out of a ~377-utility long tail, not flag composition, so
the retrieval tier needs a chunked heterogeneous corpus rather than whole
documents.  That directory measured the two corpora; this one builds the thing
it argued for.  Three chunk kinds, deliberately not equivalent:

* ``tldr_example`` — one worked example (description + command).  Runnable and
  quotable, so a span-copying model can answer from it directly.  Covers the
  head (96% of the 50 most-used utilities) and only half the tail.
* ``man_example`` — one example out of a man page EXAMPLES section.  Also
  runnable, but far rarer.
* ``man_option`` — one option plus its description.  Universal coverage, but a
  flag and a sentence is not a command: using one requires *composition*, which
  is the natural escalation boundary for a small model.

Two things about the roff side that the scoping pass got wrong by sampling, and
that this parser has to handle explicitly:

1. **`.TP` is not the universal option idiom.**  32 of the 60 man pages on this
   container carry *zero* `.TP`.  Those are the DocBook-XSL generated ones
   (PostgreSQL), which spell an option as ``.PP`` / ``\\fB\\-a\\fR`` / ``.br`` /
   ``\\fB\\-\\-data\\-only\\fR`` / ``.RS 4`` … ``.RE``.  A `.TP`-only chunker
   silently drops half the corpus, so both forms are parsed and both emit
   ``kind="man_option"``.
2. **EXAMPLES sections come in two dialects.**  pandoc-generated pages (the JDK
   ones) use ``.IP \\[bu] 2`` bullets with the command in a nested ``.RS``;
   DocBook pages use prose followed by an ``.nf`` … ``.fi`` display block.  The
   splitter handles both, preferring bullets when present.

Caveat carried into the output, not silently fixed: PostgreSQL EXAMPLES are
mostly *psql session transcripts* — SQL, not shell command lines — so
``man_example`` chunks are labelled ``runnable=true`` by kind while the summary
separately reports how many actually begin with a shell invocation of the
utility.  Believe the second number.

Utility naming: for tldr the utility is the **leading token of the command**
(wrappers like ``sudo``/``xargs`` skipped), not the page stem, so that
``git-checkout.md`` lands under ``git`` — the same extraction
``nl2sh-scoping/utility_distribution.py`` applies to NL2Bash, which is what a
downstream shortlister has to match against.  The page stem is preserved in the
chunk id.  For man pages the utility is the filename stem.

Usage::

    python3 build_corpus.py --tldr <tldr>/pages --man-dir /usr/share/man/man1

Writes ``data/chunks.jsonl`` and ``data/utilities.json`` (both gitignored;
the whole build is ~1 s, so regenerate rather than committing 6 MB of derived
data).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Command wrappers that are not the utility being documented.  Same list as
# nl2sh-scoping/utility_distribution.py, plus the ones tldr pages lead with.
WRAPPERS = {"sudo", "time", "nohup", "command", "env", "doas", "!"}
UTIL_OK = re.compile(r"[a-zA-Z0-9_.+-]+")


# ---------------------------------------------------------------- tldr ------

# "- [c]reate an archive and write it to a [f]ile:"
TLDR_DESC = re.compile(r"^- (.+?):?\s*$")
# "`tar cf {{path/to/target.tar}} ...`"  — whole line is one backticked command
TLDR_CMD = re.compile(r"^`(.+)`\s*$")
# tldr mnemonic brackets: "[c]reate" marks which letter maps to a flag
MNEMONIC = re.compile(r"\[([A-Za-z0-9])\]")

SKIPPED: collections.Counter = collections.Counter()


def tldr_placeholders(cmd: str) -> str:
    """Unwrap tldr ``{{...}}`` slots into a runnable command template.

    ``{{path/to/file}}`` -> ``path/to/file``.  ``{{[-a|--all]}}`` is an
    alternation of equivalent spellings; the first (short) form is kept so the
    result stays a single valid command.  The long spellings are not lost from
    the corpus — ``man_option`` chunks are keyed by exactly those.
    """

    def sub(m: re.Match) -> str:
        v = m.group(1)
        if v.startswith("[") and v.endswith("]"):
            return v[1:-1].split("|")[0]
        return v

    return re.sub(r"\{\{(.*?)\}\}", sub, cmd)


def leading_utility(cmd: str) -> str:
    """First real utility token of a command, skipping wrappers and VAR=x."""
    for tok in cmd.split():
        if tok in WRAPPERS:
            continue
        if "=" in tok and not tok.startswith("-"):
            continue
        tok = tok.strip("()`$\"'")
        return tok if UTIL_OK.fullmatch(tok) else ""
    return ""


def _tokens(cmd: str) -> set[str]:
    return {t.strip("()`$\"';|&") for t in cmd.split()}


def tldr_utility(stem: str, cmd: str) -> str:
    """Which utility does this tldr example belong to?

    The page stem wins whenever it actually appears as a token in the command.
    That is what makes ``sudo.md`` — whose examples all read ``sudo <something>``
    — come out as ``sudo`` rather than as whatever it happens to wrap, and it
    rescues ``color=$(hyprpicker -f hex)`` from being filed under ``-f``.

    Otherwise the leading non-wrapper token wins, which is what folds
    ``git-checkout.md`` into ``git`` — the same extraction
    ``nl2sh-scoping/utility_distribution.py`` runs over NL2Bash, and therefore
    the key a downstream shortlister has to match on.  A token starting with
    ``-`` is a flag, never a utility, so the stem is the fallback.
    """
    if stem in _tokens(cmd):
        return stem
    lead = leading_utility(cmd)
    if not lead or lead.startswith("-"):
        return stem
    return lead


def parse_tldr(path: Path) -> list[dict]:
    """One chunk per example on a tldr page: description line + command line."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stem = path.stem
    platform = path.parent.name
    out: list[dict] = []
    pending, n = None, 0
    for line in lines:
        d = TLDR_DESC.match(line)
        if d:
            pending = MNEMONIC.sub(r"\1", d.group(1)).strip()
            continue
        c = TLDR_CMD.match(line)
        if c and pending is not None:
            cmd = tldr_placeholders(c.group(1)).strip()
            # tldr ships ~670 stub pages for shell builtins and aliases whose
            # only content is "View documentation for the original command" +
            # `tldr <other>`.  They are cross-references, not documentation.
            if stem != "tldr" and leading_utility(cmd) == "tldr":
                SKIPPED["tldr_stub"] += 1
                continue
            util = tldr_utility(stem, cmd)
            desc = pending[:1].upper() + pending[1:] if pending else ""
            out.append({
                "id": f"tldr:{platform}/{stem}#{n}",
                "utility": util,
                "kind": "tldr_example",
                "text": f"{desc}\n{cmd}",
                "runnable": True,
            })
            n += 1
            pending = None
    return out


# ----------------------------------------------------------------- roff -----

# Escapes, longest/most specific first.  \f[...] is the pandoc font form,
# \fX the classic one.
ROFF_ESCAPES = [
    (r"\\f\[[A-Za-z]*\]", ""),          # \f[V] \f[B] \f[R] \f[CB] \f[VB]
    (r"\\f[BIRPS]", ""),                # \fB \fI \fR \fP
    (r"\\f\(..", ""),                   # \f(CW
    (r"\\\*\(Aq", "'"), (r"\\\*\(lq", '"'), (r"\\\*\(rq", '"'),
    (r"\\\*\(dq", '"'), (r"\\\*\([A-Za-z]{2}", ""),
    (r"\\\[dq\]", '"'), (r"\\\[aq\]", "'"), (r"\\\[rs\]", "\\\\"),
    (r"\\\[bu\]", "-"), (r"\\\[ti\]", "~"), (r"\\\[ha\]", "^"),
    (r"\\\[lq\]", '"'), (r"\\\[rq\]", '"'), (r"\\\[oq\]", "'"),
    (r"\\\[cq\]", "'"), (r"\\\[em\]", "--"), (r"\\\[en\]", "-"),
    (r"\\\[[A-Za-z0-9]+\]", ""),        # any other named glyph
    (r"\\\(aq", "'"), (r"\\\(dq", '"'), (r"\\\(bu", "-"),
    (r"\\\(em", "--"), (r"\\\(en", "-"), (r"\\\(..", ""),
    (r"\\&", ""), (r"\\%", ""), (r"\\c", ""),
    (r"\\\{", ""), (r"\\\}", ""),
    (r"\\-", "-"), (r"\\e", "\\\\"), (r"\\ ", " "), (r"\\\\", "\\\\"),
    (r"\\n\(..", ""), (r"\\s[-+]?\d", ""),
]
ROFF_ESCAPES = [(re.compile(p), r) for p, r in ROFF_ESCAPES]

# Macros whose arguments ARE the text: drop the macro name, keep the rest.
TEXT_MACROS = {"B", "I", "BR", "BI", "IB", "IR", "RI", "RB", "SM", "SB"}
# Macros that are pure layout, metadata or groff flow control: drop the line
# entirely, arguments included.  `.if`/`.ie`/`.el` matter here — DocBook pages
# wrap every display block in `.if n \{\` ... `.\}`, and keeping their
# arguments leaks literal "n \{\ 4" into the chunk text.
DROP_MACROS = {
    "PP", "LP", "P", "sp", "br", "RS", "RE", "nf", "fi", "ad", "na", "nh",
    "hy", "if", "ie", "el", "ds", "de", "TH", "IP", "TP", "TS", "TE", "T&",
    "in", "ll", "PD", "ne", "nr", "rr", "so", "ft", "fam", "EX", "EE", "SS",
    "SH", "UR", "UE", "MT", "ME", "Vb", "Ve", "ce", "rs", "\\}", "\\{", "}", "{",
}
MACRO_NAME = re.compile(r"^['.](\S*)")


def deroff_line(line: str) -> str:
    """One roff source line -> readable text ('' if it carries none)."""
    line = line.rstrip()
    if not line:
        return ""
    if line.startswith('.\\"') or line.startswith("'\\\""):
        return ""
    if line[0] in ".'":
        macro = MACRO_NAME.match(line).group(1)
        if macro in TEXT_MACROS:
            line = line[1 + len(macro):]
        elif macro in DROP_MACROS or macro.startswith("\\") or not macro:
            return ""
        else:
            # unrecognised macro: keep its arguments, drop the name
            rest = line.split(None, 1)
            line = rest[1] if len(rest) > 1 else ""
    for pat, rep in ROFF_ESCAPES:
        line = pat.sub(rep, line)
    return line.strip()


def deroff(lines: list[str], keep_breaks: bool = False) -> str:
    """Block of roff source -> text.

    roff wraps prose across source lines, so prose is rejoined with spaces.
    ``keep_breaks`` preserves one line per source line, which is what a
    literal (``.nf``) display block needs.
    """
    parts = [deroff_line(l) for l in lines]
    parts = [p for p in parts if p]
    if keep_breaks:
        return "\n".join(parts)
    return re.sub(r"\s{2,}", " ", " ".join(parts)).strip()


def read_man(path: Path) -> list[str]:
    with gzip.open(path, "rt", errors="replace") as fh:
        return fh.read().splitlines()


def sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split a man page into (SECTION NAME, body lines) at .SH."""
    out, name, body = [], None, []
    for line in lines:
        m = re.match(r'^\.SH\s+"?([^"]*)"?\s*$', line)
        if m:
            if name is not None:
                out.append((name, body))
            name, body = m.group(1).strip(), []
        elif name is not None:
            body.append(line)
    if name is not None:
        out.append((name, body))
    return out


OPT_TAG = re.compile(r"^\s*-{1,2}[A-Za-z0-9]")


def man_options(utility: str, page: list[str]) -> list[dict]:
    """Option-plus-description blocks, in both roff dialects.

    Form A (classic/pandoc): ``.TP`` / tag line / body until the next ``.TP``,
    ``.SH`` or ``.SS``.

    Form B (DocBook-XSL, e.g. PostgreSQL): ``.PP`` / one or more ``\\fB-x\\fR``
    tag lines separated by ``.br`` / ``.RS 4`` body ``.RE``.  32 of the 60 man
    pages here have no ``.TP`` at all and would otherwise contribute nothing.
    """
    out: list[dict] = []
    n = 0

    for sec_name, body in sections(page):
        # ---- Form A: .TP blocks
        idx = [i for i, l in enumerate(body) if l.strip() == ".TP" or l.startswith(".TP ")]
        for i in idx:
            j = i + 1
            while j < len(body) and not (
                body[j].strip() == ".TP" or body[j].startswith(".TP ")
                or body[j].startswith(".SS") or body[j].startswith(".SH")
            ):
                j += 1
            block = body[i + 1:j]
            if not block:
                continue
            tag = deroff(block[:1])
            desc = deroff(block[1:])
            if not tag or not desc:
                continue
            if not OPT_TAG.match(tag) and "OPTION" not in sec_name.upper():
                continue
            out.append({
                "id": f"man:{utility}#opt{n}", "utility": utility,
                "kind": "man_option", "text": f"{tag}\n{desc}",
                "runnable": False,
            })
            n += 1

        # ---- Form B: .PP <tag lines> .RS 4 ... .RE
        i = 0
        while i < len(body):
            if body[i].startswith(".PP"):
                j = i + 1
                tags = []
                while j < len(body) and (
                    body[j].startswith("\\fB") or body[j].startswith("\\fI")
                    or body[j].startswith(".br")
                ):
                    if not body[j].startswith(".br"):
                        tags.append(body[j])
                    j += 1
                if tags and j < len(body) and body[j].startswith(".RS"):
                    depth, k = 1, j + 1
                    while k < len(body) and depth:
                        if body[k].startswith(".RS"):
                            depth += 1
                        elif body[k].startswith(".RE"):
                            depth -= 1
                        k += 1
                    tag = " ".join(deroff([t]) for t in tags).strip()
                    desc = deroff(body[j + 1:k - 1])
                    if tag and desc and (
                        OPT_TAG.match(tag) or "OPTION" in sec_name.upper()
                    ):
                        out.append({
                            "id": f"man:{utility}#opt{n}", "utility": utility,
                            "kind": "man_option", "text": f"{tag}\n{desc}",
                            "runnable": False,
                        })
                        n += 1
                    i = k
                    continue
            i += 1
    return out


def _display_blocks(body: list[str]) -> list[tuple[int, int]]:
    """(start, end) index spans of .nf ... .fi literal display blocks."""
    spans, start = [], None
    for i, l in enumerate(body):
        if l.startswith(".nf"):
            start = i
        elif l.startswith(".fi") and start is not None:
            spans.append((start, i))
            start = None
    return spans


def man_examples(utility: str, page: list[str]) -> list[dict]:
    """Split an EXAMPLES section into one chunk per example.

    Three dialects observed on this container's 60 pages:

    * **bulleted** (pandoc/JDK, e.g. ``jar``) — each ``.IP \\[bu]`` starts an
      example; prose first, command in a nested ``.RS``.
    * **display** (DocBook/PostgreSQL, e.g. ``psql``) — prose accumulates until
      an ``.nf`` … ``.fi`` literal block closes; prose + literal is one example.
    * **heading-plus-display** (``jpackage``) — alternating display blocks where
      every other one is a single line ending in ``:`` acting as a heading.
      Those are merged forward into the block they introduce, otherwise the
      corpus gets a chunk that is a title with no command under it.

    A segment with no literal/display part is prose, not an example, and is
    dropped — that is where the commands live in all three dialects.
    """
    out: list[dict] = []
    n = 0
    for sec_name, body in sections(page):
        if "EXAMPLE" not in sec_name.upper():
            continue

        bullets = [i for i, l in enumerate(body) if l.startswith(".IP \\[bu]")]
        segments: list[list[str]] = []
        if bullets:
            for a, b in zip(bullets, bullets[1:] + [len(body)]):
                segments.append(body[a + 1:b])
        else:
            prev = 0
            for a, b in _display_blocks(body):
                segments.append(body[prev:b + 1])
                prev = b + 1

        parsed: list[tuple[str, str]] = []   # (prose, literal)
        for seg in segments:
            spans = _display_blocks(seg)
            if spans:
                lit_lines: list[str] = []
                prose_lines = list(seg)
                for a, b in reversed(spans):
                    lit_lines = seg[a + 1:b] + lit_lines
                    del prose_lines[a:b + 1]
                parsed.append((deroff(prose_lines), deroff(lit_lines, keep_breaks=True)))
            else:
                # bulleted dialect: the command sits in a nested .RS block
                rs = next((i for i, l in enumerate(seg) if l.startswith(".RS")), None)
                if rs is None:
                    parsed.append((deroff(seg), ""))
                else:
                    parsed.append((deroff(seg[:rs]), deroff(seg[rs:], keep_breaks=True)))

        # merge a heading-only display block into the one that follows it
        merged: list[tuple[str, str]] = []
        carry = ""
        for prose, literal in parsed:
            if literal and "\n" not in literal and literal.rstrip().endswith(":"):
                carry = " ".join(x for x in (carry, prose, literal) if x).strip()
                continue
            merged.append((" ".join(x for x in (carry, prose) if x).strip(), literal))
            carry = ""

        for prose, literal in merged:
            if not literal:
                continue        # prose without a display block is not an example
            text = "\n".join(x for x in (prose, literal) if x).strip()
            out.append({
                "id": f"man:{utility}#ex{n}", "utility": utility,
                "kind": "man_example", "text": text, "runnable": True,
            })
            n += 1
    return out


# ------------------------------------------------------------- reporting ----


def approx_tokens(text: str) -> int:
    return len(text) // 4


def pct(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tldr", type=Path, required=True,
                    help="tldr checkout's pages/ directory")
    ap.add_argument("--platforms", nargs="+", default=["common", "linux", "osx"])
    ap.add_argument("--man-dir", type=Path, default=Path("/usr/share/man/man1"))
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []

    tldr_files = []
    for plat in a.platforms:
        tldr_files += sorted((a.tldr / plat).glob("*.md"))
    for f in tldr_files:
        chunks += parse_tldr(f)
    n_tldr_pages = len(tldr_files)

    man_files = sorted(a.man_dir.glob("*.gz"))
    pages_with_examples = 0
    pages_with_tp = 0
    for f in man_files:
        utility = f.name.split(".")[0]
        page = read_man(f)
        if any(l.strip() == ".TP" or l.startswith(".TP ") for l in page):
            pages_with_tp += 1
        opts = man_options(utility, page)
        exs = man_examples(utility, page)
        if exs:
            pages_with_examples += 1
        chunks += opts + exs

    # Deduplicate on (utility, kind, text).  Two sources of exact duplicates:
    # a tldr example repeated across platform directories (common + linux/osx),
    # and a handful of man options matched by both the .TP and the DocBook
    # parser on pages that mix idioms.  Duplicate chunks waste top-k slots.
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for c in chunks:
        key = (c["utility"], c["kind"], c["text"])
        if key in seen:
            SKIPPED["duplicate"] += 1
            continue
        seen.add(key)
        deduped.append(c)
    chunks = deduped

    out_path = a.out_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- utilities.json
    utils: dict[str, dict] = {}
    for c in chunks:
        u = utils.setdefault(c["utility"], {"kinds": {}, "n_chunks": 0,
                                            "has_runnable": False})
        u["kinds"][c["kind"]] = u["kinds"].get(c["kind"], 0) + 1
        u["n_chunks"] += 1
        u["has_runnable"] |= bool(c["runnable"])
    for u in utils.values():
        u["kinds"] = dict(sorted(u["kinds"].items()))
    (a.out_dir / "utilities.json").write_text(
        json.dumps(dict(sorted(utils.items())), indent=1, ensure_ascii=False) + "\n")

    # ---- report
    by_kind = collections.Counter(c["kind"] for c in chunks)
    toks = [approx_tokens(c["text"]) for c in chunks]
    print(f"skipped: {SKIPPED['tldr_stub']} tldr stub/redirect examples, "
          f"{SKIPPED['duplicate']} exact duplicates")
    print(f"tldr pages read: {n_tldr_pages}   man pages read: {len(man_files)} "
          f"({pages_with_tp} with .TP, {pages_with_examples} yielding examples)")
    print(f"\nchunks: {len(chunks)}")
    for k in ("tldr_example", "man_option", "man_example"):
        ks = [c for c in chunks if c["kind"] == k]
        kt = [approx_tokens(c["text"]) for c in ks]
        print(f"  {k:<14} {by_kind[k]:>7}   utilities {len({c['utility'] for c in ks}):>5}"
              f"   tokens median {statistics.median(kt) if kt else 0:>5.0f}"
              f"  p90 {pct(kt, 0.9):>6.0f}  max {max(kt) if kt else 0:>6}")
    print(f"\ndistinct utilities: {len(utils)}")
    print(f"  with >=1 runnable chunk: {sum(u['has_runnable'] for u in utils.values())}")
    print(f"  tldr only : {sum(set(u['kinds']) == {'tldr_example'} for u in utils.values())}")
    print(f"  man only  : {sum('tldr_example' not in u['kinds'] for u in utils.values())}")
    print(f"  both      : {sum('tldr_example' in u['kinds'] and len(u['kinds']) > 1 for u in utils.values())}")
    print(f"\nchunk tokens (chars/4), all kinds: median {statistics.median(toks):.0f} "
          f" p90 {pct(toks, 0.9):.0f}  max {max(toks)}")
    print(f"  <=350 tokens: {sum(t <= 350 for t in toks) / len(toks):.1%}")

    # Honesty check on the man_example runnable=true label: how many actually
    # open with a shell invocation of the utility?  PostgreSQL EXAMPLES are
    # largely psql session transcripts (SQL), not shell command lines.
    shellish = 0
    mex = [c for c in chunks if c["kind"] == "man_example"]
    for c in mex:
        if any((l.strip().lstrip("$#%> ").split() or [""])[0] == c["utility"]
               for l in c["text"].splitlines()):
            shellish += 1
    if mex:
        print(f"\nman_example chunks containing a line that invokes the utility: "
              f"{shellish}/{len(mex)} ({shellish / len(mex):.0%})")
    print(f"\nwrote {out_path} and {a.out_dir / 'utilities.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
