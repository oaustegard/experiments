#!/usr/bin/env python3
"""Reconstruct full post text from the embedding muninn.kb's own chunks.

Concatenating each post's chunks (in file order) yields text byte-identical to
what the embedding KB indexed, so a lexical re-chunk at any size compares against
embeddings with chunk-size as the *only* variable (same extraction, same text).

Writes one file per post under OUT/<section>/<slug>.txt (preserving source_path
so the lexical builder derives matching section/source_path meta), and prints the
post inventory (source_path + title) for gold-label selection.
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke  # noqa: E402

HERE = Path(__file__).resolve().parent
KB = spoke("muninn.austegard.com") / "knowledge/muninn.kb"
OUT = HERE / "corpus"


def main() -> int:
    z = zipfile.ZipFile(KB)
    chunks = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    # group chunks by source_path, preserving order
    posts: "OrderedDict[str, list]" = OrderedDict()
    meta_by_post = {}
    for c in chunks:
        sp = c["meta"]["source_path"]
        posts.setdefault(sp, []).append(c["text"])
        meta_by_post.setdefault(sp, c["meta"])

    OUT.mkdir(parents=True, exist_ok=True)
    for sp, texts in posts.items():
        body = "\n\n".join(texts)
        dest = OUT / sp.replace(".html", ".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")

    print(f"reconstructed {len(posts)} posts ({len(chunks)} chunks) -> {OUT}")
    print("\nsource_path\ttitle")
    for sp, m in meta_by_post.items():
        print(f"{sp}\t{m.get('title','')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
