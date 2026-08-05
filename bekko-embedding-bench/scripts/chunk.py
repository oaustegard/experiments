"""Two chunkings of a source tree, varying exactly one axis.

- ``ast``  — function/class granularity via stdlib ``ast``, with the enclosing
  class name and the def signature prepended to each chunk body. Non-Python
  sources (.pyx/.pxd/.pyx.tp) have no stdlib parser, so they fall back to
  line-windows; the file is recorded either way so recall is comparable.
- ``flat`` — fixed line-windows with overlap, no structure at all.

The point of the pair is that chunk boundaries are a confound when comparing
encoders; holding the encoder fixed and swapping only this isolates it.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

EXTS = (".py", ".pyx", ".pxd", ".pyx.tp", ".pxi")
SKIP_DIRS = {".git", "build", "doc", "__pycache__", ".mypy_cache"}

WINDOW = 60
STRIDE = 45
MIN_CHARS = 80
MAX_CHARS = 2400


def iter_files(root: Path, subdir: str = "sklearn") -> list[Path]:
    out = []
    for p in sorted((root / subdir).rglob("*")):
        if not p.is_file():
            continue
        if any(d in p.parts for d in SKIP_DIRS):
            continue
        if p.name.endswith(EXTS):
            out.append(p)
    return out


def _clip(s: str) -> str:
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS]


def line_windows(rel: str, text: str) -> list[dict]:
    lines = text.split("\n")
    out = []
    for s in range(0, max(1, len(lines)), STRIDE):
        body = "\n".join(lines[s : s + WINDOW])
        if len(body.strip()) < MIN_CHARS:
            continue
        out.append({"file": rel, "start": s + 1, "text": _clip(body)})
        if s + WINDOW >= len(lines):
            break
    return out


def ast_chunks(rel: str, text: str) -> list[dict]:
    """Function/class-scoped chunks with signature + enclosing class prepended."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return line_windows(rel, text)
    lines = text.split("\n")
    out: list[dict] = []

    def emit(node, cls: str | None) -> None:
        s = node.lineno - 1
        e = getattr(node, "end_lineno", s + 1)
        body = "\n".join(lines[s:e])
        if len(body.strip()) < MIN_CHARS:
            return
        # Header carries the identifiers a reader would use to recognize this
        # chunk: module path, enclosing class, and the def line itself.
        head = f"# {rel}"
        if cls:
            head += f" | class {cls}"
        head += f"\n{lines[s].strip()}"
        out.append({"file": rel, "start": node.lineno, "text": _clip(head + "\n" + body)})

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, None)
        elif isinstance(node, ast.ClassDef):
            methods = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not methods:
                emit(node, None)
                continue
            # class-level docstring/attribute block as its own chunk
            head_end = methods[0].lineno - 1
            block = "\n".join(lines[node.lineno - 1 : head_end])
            if len(block.strip()) >= MIN_CHARS:
                out.append(
                    {"file": rel, "start": node.lineno, "text": _clip(f"# {rel}\n" + block)}
                )
            for m in methods:
                emit(m, node.name)
    if not out:
        return line_windows(rel, text)
    return out


def build(root: Path, mode: str, subdir: str = "sklearn") -> list[dict]:
    chunks = []
    for p in iter_files(root, subdir):
        rel = str(p.relative_to(root))
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        if mode == "ast" and p.name.endswith(".py"):
            chunks += ast_chunks(rel, text)
        else:
            chunks += line_windows(rel, text)
    return chunks


if __name__ == "__main__":
    import json
    import sys

    root = Path(os.environ.get("BEKKO_BENCH_REPO", "/home/user/sklearn-bench"))
    for mode in ("ast", "flat"):
        c = build(root, mode)
        json.dump(c, open(f"chunks_{mode}.json", "w"))
        nf = len({x["file"] for x in c})
        avg = sum(len(x["text"]) for x in c) / max(1, len(c))
        print(f"{mode}: {len(c)} chunks over {nf} files, mean {avg:.0f} chars", file=sys.stderr)
