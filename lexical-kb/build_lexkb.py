#!/usr/bin/env python3
"""Build a portable, embedding-free lexical KB `.skill` bundle.

Pipeline: collect files -> structural chunk -> BM25 inverted index -> write a
bundle dir (chunks.jsonl + index.json + SKILL.md + search.py) -> optionally zip
to `<name>.skill` (an ordinary zip with a renamed extension).

No embeddings, no model, no third-party deps. The shipped `search.py` owns the
tokenizer; this builder imports it so the index and queries tokenize identically.

Usage:
    python build_lexkb.py CORPUS_DIR --out out/mykb --name mykb \\
        --target-chars 1200 [--zip] [--ext txt,md,html]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Import the shipped tokenizer so builder/searcher cannot diverge.
sys.path.insert(0, str(Path(__file__).resolve().parent / "skill_template"))
from search import tokenize  # noqa: E402

_TEMPLATE_DIR = Path(__file__).resolve().parent / "skill_template"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


# --------------------------------------------------------------------------- #
# Text extraction + structural chunking
# --------------------------------------------------------------------------- #


def extract_text(path: Path) -> tuple[str, str]:
    """Return (title, body_text). HTML is crudely de-tagged (no bs4 dep)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        m = re.search(r"<title>(.*?)</title>", raw, re.I | re.S)
        title = (m.group(1).strip() if m else path.stem)
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
        body = _TAG_RE.sub("\n", body)
        body = re.sub(r"&[a-zA-Z#0-9]+;", " ", body)
    else:
        title = path.stem
        body = raw
        # first non-empty line doubles as a human title for txt/md (capped)
        for line in raw.splitlines():
            if line.strip():
                t = line.strip().lstrip("# ").strip()
                title = (t[:80] + "…") if len(t) > 80 else t
                break
    # normalize whitespace but keep paragraph breaks
    lines = [_WS_RE.sub(" ", ln).rstrip() for ln in body.splitlines()]
    return title, "\n".join(lines)


def chunk_text(text: str, target_chars: int) -> list[str]:
    """Greedy structural chunking: pack whole paragraphs up to target_chars.
    A single oversized paragraph becomes its own chunk (never split mid-para).
    target_chars <= 0 means whole-document (one chunk)."""
    text = text.strip()
    if not text:
        return []
    if target_chars <= 0:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= target_chars:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def collect_chunks(corpus: Path, exts: set[str], target_chars: int, min_chars: int) -> list[dict]:
    files = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in exts)
    chunks: list[dict] = []
    for path in files:
        rel = path.relative_to(corpus).as_posix()
        title, text = extract_text(path)
        pieces = chunk_text(text, target_chars)
        for j, piece in enumerate(pieces):
            if len(piece) < min_chars:
                continue
            chunks.append({
                "id": f"{rel}#chunk-{j}",
                "text": piece,
                "meta": {
                    "title": title,
                    "source_path": rel,
                    "section": rel.split("/", 1)[0] if "/" in rel else "",
                },
            })
    return chunks


# --------------------------------------------------------------------------- #
# BM25 inverted index
# --------------------------------------------------------------------------- #


def build_index(chunks: list[dict], k1: float, b: float) -> dict:
    postings: dict[str, list[list[int]]] = defaultdict(list)
    doclen: list[int] = []
    for i, ch in enumerate(chunks):
        toks = tokenize(ch["text"])
        doclen.append(len(toks))
        for term, c in Counter(toks).items():
            postings[term].append([i, c])
    N = len(chunks)
    avgdl = (sum(doclen) / N) if N else 0.0
    return {
        "params": {"k1": k1, "b": b},
        "N": N,
        "avgdl": avgdl,
        "doclen": doclen,
        "df": {t: len(pl) for t, pl in postings.items()},
        "postings": dict(postings),
    }


# --------------------------------------------------------------------------- #
# Bundle writer
# --------------------------------------------------------------------------- #


def write_bundle(out_dir: Path, chunks: list[dict], index: dict, source_desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
    with (out_dir / "index.json").open("w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    shutil.copy2(_TEMPLATE_DIR / "search.py", out_dir / "search.py")
    skill_md = (_TEMPLATE_DIR / "SKILL.md").read_text(encoding="utf-8")
    skill_md = skill_md.replace("{{SOURCE}}", source_desc).replace("{{CHUNK_COUNT}}", str(len(chunks)))
    (out_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")


def zip_skill(out_dir: Path, skill_path: Path) -> None:
    """An ordinary zip with a `.skill` extension. The bundle dir's contents are
    placed under a top-level folder named after the skill (skill convention)."""
    root = skill_path.stem
    with zipfile.ZipFile(skill_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                zf.write(f, f"{root}/{f.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="directory of source files")
    ap.add_argument("--out", default="out/kb", help="bundle output directory")
    ap.add_argument("--name", default="lexical-kb", help="skill name (zip root + .skill stem)")
    ap.add_argument("--ext", default="txt,md,html,htm", help="comma-separated extensions to index")
    ap.add_argument("--target-chars", type=int, default=1200, help="chunk size target (<=0 = whole document)")
    ap.add_argument("--min-chars", type=int, default=40, help="drop chunks shorter than this")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--source", default="", help="human description of the corpus")
    ap.add_argument("--zip", action="store_true", help="also emit <name>.skill zip next to --out")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    exts = {e.strip().lstrip(".").lower() for e in args.ext.split(",") if e.strip()}
    chunks = collect_chunks(corpus, exts, args.target_chars, args.min_chars)
    if not chunks:
        print(f"no chunks produced from {corpus} (exts={exts})", file=sys.stderr)
        return 1
    index = build_index(chunks, args.k1, args.b)

    out_dir = Path(args.out)
    write_bundle(out_dir, chunks, index, args.source or f"{corpus.name} corpus")

    n_files = len({c["meta"]["source_path"] for c in chunks})
    avg = index["avgdl"]
    print(f"built {len(chunks)} chunks from {n_files} files "
          f"(target_chars={args.target_chars}, avgdl={avg:.0f} tokens, "
          f"vocab={len(index['df'])}) -> {out_dir}")

    if args.zip:
        skill_path = out_dir.parent / f"{args.name}.skill"
        zip_skill(out_dir, skill_path)
        print(f"zipped -> {skill_path} ({skill_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
