#!/usr/bin/env python3
"""Which documentation corpus should the retrieval tier read — tldr or man?

An earlier draft of this directory's README answered "tldr", on the grounds that
72.6% of commands carry at most one flag and a whole man page is enormous. That
argument conflated *RAG over man pages* with *putting a man page in the context
window*. They are not the same thing: retrieval chunks, and a man page chunks
extremely well. This measures both corpora properly.

Two questions, two measurements:

* **Coverage** — of the utilities that actually appear in NL2Bash, how many have
  a tldr page? Reported for the head and for the used-once tail separately,
  because the tail is where the requests are (the top 10 utilities cover only
  29.1% of non-find requests) and where a model most needs help.
* **Chunkability** — roff gives man pages free, self-delimiting chunk
  boundaries: `.SH` for sections and `.TP` for one option plus its description.
  Measured as approximate tokens (chars/4).

A caveat that bit the first run: **`man -w` is useless as a coverage probe on a
minimised container.** Debian's stub exits 0 and prints "This system has been
minimized" for *any* argument, including a nonsense one, which produced a
spurious 100%. Man coverage is measured off the filesystem, and the honest
statement is structural rather than numeric — a man page exists for what is
installed.

    git clone --depth 1 https://github.com/TellinaTool/nl2bash.git
    git clone --depth 1 --filter=blob:none --sparse https://github.com/tldr-pages/tldr.git
    cd tldr && git sparse-checkout set pages && cd ..
    python3 doc_corpus.py --nl2bash nl2bash/data/bash/all.cm --tldr tldr/pages
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
SPLIT = re.compile(r"\|\||&&|\||;")
WRAPPERS = {"sudo", "time", "nohup", "command", "!"}
UTIL_OK = re.compile(r"[a-z0-9_.+-]+")


def _mask(c: str) -> str:
    return re.sub(r'"[^"]*"|\'[^\']*\'', '""', c)


def utility(segment: str) -> str:
    for tok in segment.split():
        if tok in WRAPPERS:
            continue
        if "=" in tok and not tok.startswith("-"):
            continue
        return tok.strip("()`$")
    return ""


def utilities(cmds: list[str]) -> collections.Counter:
    """Every utility used anywhere in a command, not just the leading one."""
    out: collections.Counter = collections.Counter()
    for c in cmds:
        for seg in SPLIT.split(_mask(c)):
            u = utility(seg)
            if u and UTIL_OK.fullmatch(u):
                out[u] += 1
    return out


def man_stats(man_dir: Path) -> dict:
    pages = sorted(man_dir.glob("*.gz"))
    whole, tp, with_examples = [], [], 0
    for p in pages:
        try:
            t = gzip.open(p, "rt", errors="replace").read()
        except Exception:
            continue
        whole.append(len(t) // 4)
        if any("EXAMPLE" in s for s in re.findall(r'^\.SH\s+"?([A-Z ]+)"?', t, re.M)):
            with_examples += 1
        for block in re.split(r"^\.TP\b", t, flags=re.M)[1:]:
            tp.append(len(block.split("\n.TP")[0]) // 4)
    q = lambda xs, i: statistics.quantiles(xs, n=10)[i] if len(xs) > 10 else max(xs)
    return {
        "n_pages": len(whole), "n_option_chunks": len(tp),
        "pages_with_examples": with_examples,
        "whole_page_tokens": {"median": statistics.median(whole), "p90": round(q(whole, 8)),
                              "max": max(whole)},
        "option_chunk_tokens": {"median": statistics.median(tp), "p90": round(q(tp, 8)),
                                "max": max(tp)},
        "option_chunks_under_350_tok": round(sum(c <= 350 for c in tp) / len(tp), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl2bash", type=Path, required=True)
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--man-dir", type=Path, default=Path("/usr/share/man/man1"))
    ap.add_argument("--out", type=Path, default=HERE / "doc_corpus.json")
    a = ap.parse_args()

    cmds = [l for l in a.nl2bash.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    used = utilities(cmds)
    tldr = {p.stem for p in a.tldr.rglob("*.md")}

    slices = {"all": list(used),
              "top 50 by usage": [u for u, _ in used.most_common(50)],
              "top 200 by usage": [u for u, _ in used.most_common(200)],
              "used exactly once": [u for u, c in used.items() if c == 1]}
    cov = {k: {"n": len(v), "tldr": len([u for u in v if u in tldr]),
               "tldr_frac": round(len([u for u in v if u in tldr]) / len(v), 4)}
           for k, v in slices.items()}

    out = {"n_tldr_pages": len(tldr), "n_utilities_in_nl2bash": len(used),
           "tldr_coverage": cov, "man": man_stats(a.man_dir)}
    a.out.write_text(json.dumps(out, indent=1) + "\n")

    print(f"tldr corpus: {len(tldr)} pages;  NL2Bash uses {len(used)} distinct utilities\n")
    for k, v in cov.items():
        print(f"  {k:<20} n={v['n']:>4}   tldr covers {v['tldr']:>4}   {v['tldr_frac']:6.1%}")
    m = out["man"]
    print(f"\nman pages sampled: {m['n_pages']}  ({m['pages_with_examples']} with an EXAMPLES section)")
    print(f"  whole page tokens   median {m['whole_page_tokens']['median']:>6.0f}"
          f"   p90 {m['whole_page_tokens']['p90']:>6}   max {m['whole_page_tokens']['max']:>6}")
    print(f"  .TP option chunk    median {m['option_chunk_tokens']['median']:>6.0f}"
          f"   p90 {m['option_chunk_tokens']['p90']:>6}   n={m['n_option_chunks']}")
    print(f"  chunks <=350 tokens {m['option_chunks_under_350_tok']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
