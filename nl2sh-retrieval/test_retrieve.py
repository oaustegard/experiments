#!/usr/bin/env python3
"""Sanity tests for the code-aware tokenizer and the BM25 index.

No network, no corpus: the index is built from a handful of literal chunks so
the assertions are about behaviour, not about whatever happens to be in
data/chunks.jsonl. Run: python3 test_retrieve.py
"""
from __future__ import annotations

from retrieve import Chunk, Index, tokens


def test_compound_emits_whole_and_parts():
    assert tokens("--max-depth") == ["max-depth", "max", "depth"]
    assert tokens("git-checkout") == ["git-checkout", "git", "checkout"]
    assert tokens("NO_COLOR") == ["no_color", "no", "color"]
    assert tokens("asciiFold") == ["asciifold", "ascii", "fold"]
    # A single-part token is emitted once, not duplicated.
    assert tokens("grep") == ["grep"]


def test_stopwords_and_length_cap():
    assert tokens("how do i list the files") == ["list", "files"]
    assert tokens("a" * 41) == []


def test_stemmer_is_conservative():
    assert tokens("files", stem=True) == ["file"]
    assert tokens("listing", stem=True) == ["list"]
    assert tokens("--max-depth", stem=True)[0] == "max-depth"  # flags untouched


def _idx():
    return Index([
        Chunk("a", "find", "tldr_example",
              "Find files by name, limiting recursion depth\nfind . --max-depth 2 -name '*.py'", True),
        Chunk("b", "tar", "tldr_example", "Extract an archive\ntar -xzvf archive.tar.gz", True),
        Chunk("c", "grep", "man_option", "-r, --recursive  read all files under each directory", False),
    ])


def test_exact_flag_query_ranks_owning_chunk_first():
    hits = _idx().search("--max-depth", k=3)
    assert hits[0][0].utility == "find"


def test_prose_query_reaches_the_compound_chunk():
    # No literal "--max-depth" in the query: only the split part "depth" can
    # match, which a tokenizer keeping compounds whole would miss entirely.
    utils = _idx().topk_utilities("what limits the depth", k=3)
    assert utils == ["find"]


def test_morphology_is_not_handled_without_stemming():
    # Documented limitation, not an accident: "recurse" never reaches
    # "--recursive" unless the index is built with stem=True.
    assert _idx().topk("recurse", 3) == []


def test_utility_name_is_searchable_even_when_absent_from_text():
    assert _idx().topk_utilities("grep", k=1) == ["grep"]


def test_zero_score_returns_nothing():
    assert _idx().topk("quantum chromodynamics", 5) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
