"""Build the four arms' chunk sets from the /mnt/skills/user SKILL.md corpus.

Arms A, B0 and B1 share byte-identical chunk boundaries and bodies; they differ
only in what is prepended to the indexed text. B2 additionally forbids a chunk
from spanning a heading.

`tracked_chunker` reimplements `remax_kb.pack.default_chunker` while carrying the
heading path in effect at each chunk's first sentence. `verify_parity` asserts
its bodies are byte-identical to the real `default_chunker`, so arm A is exactly
current production behaviour rather than an approximation of it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/remax_kb")

from remax_kb.handlers import handle_markdown
from remax_kb.pack import Chunk, default_chunker

SKILLS_ROOT = Path("/mnt/skills/user")
TARGET_CHARS = 500
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _sentences(paragraph: str) -> list[str]:
    """Exactly default_chunker's sentence split: keep delimiters, flush tail."""
    pieces: list[str] = []
    cur = ""
    for ch in paragraph:
        cur += ch
        if ch in ".!?":
            pieces.append(cur)
            cur = ""
    if cur:
        pieces.append(cur)
    return pieces


def tracked_chunker(
    text: str, *, source_path: str, target_chars: int = TARGET_CHARS
) -> list[tuple[str, str, str]]:
    """default_chunker's algorithm, plus the heading path per chunk.

    Returns (chunk_id, body, heading_path). The heading path is the stack of
    markdown headings in effect when the chunk's FIRST sentence was buffered,
    joined with ' > '. Headings themselves stay in the body, exactly as
    default_chunker leaves them.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[tuple[str, str, str]] = []
    buf = ""
    stack: dict[int, str] = {}
    buf_path = ""  # heading path captured when buf started

    def path_now() -> str:
        return " > ".join(stack[lvl] for lvl in sorted(stack))

    def flush():
        nonlocal buf
        if buf.strip():
            out.append((f"{source_path}#chunk-{len(out):04d}", buf.strip(), buf_path))
        buf = ""

    for para in paragraphs:
        # A paragraph may be a heading line (possibly several stacked lines).
        for line in para.split("\n"):
            m = HEADING_RE.match(line.strip())
            if m:
                lvl = len(m.group(1))
                stack[lvl] = m.group(2).strip()
                for deeper in [k for k in stack if k > lvl]:
                    del stack[deeper]

        for s in _sentences(para):
            if not buf:
                buf = s.strip()
                buf_path = path_now()
            elif len(buf) + len(s) + 1 <= target_chars:
                buf = f"{buf} {s.strip()}"
            else:
                flush()
                buf = s.strip()
                buf_path = path_now()
        if len(buf) >= target_chars * 0.6:
            flush()
    flush()
    return out


def section_chunker(
    text: str, *, source_path: str, target_chars: int = TARGET_CHARS
) -> list[tuple[str, str, str]]:
    """Arm B2: same packing, but a chunk never spans a heading boundary."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    stack: dict[int, str] = {}
    cur: list[str] = []

    def path_now() -> str:
        return " > ".join(stack[lvl] for lvl in sorted(stack))

    cur_path = ""
    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            if cur:
                sections.append((cur_path, cur))
                cur = []
            lvl = len(m.group(1))
            stack[lvl] = m.group(2).strip()
            for deeper in [k for k in stack if k > lvl]:
                del stack[deeper]
            cur_path = path_now()
            cur = [line]
        else:
            cur.append(line)
    if cur:
        sections.append((cur_path, cur))

    out: list[tuple[str, str, str]] = []
    for path, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for ch in default_chunker(body, source_path=source_path, target_chars=target_chars):
            out.append((f"{source_path}#chunk-{len(out):04d}", ch.text, path))
    return out


def verify_parity(text: str, source_path: str) -> None:
    """tracked_chunker must reproduce default_chunker's bodies byte for byte."""
    mine = [body for _, body, _ in tracked_chunker(text, source_path=source_path)]
    theirs = [c.text for c in default_chunker(text, source_path=source_path,
                                              target_chars=TARGET_CHARS)]
    if mine != theirs:
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                raise AssertionError(
                    f"{source_path} chunk {i} differs:\n  mine={a[:120]!r}\n  theirs={b[:120]!r}"
                )
        raise AssertionError(
            f"{source_path}: chunk count {len(mine)} vs {len(theirs)}"
        )


def build() -> dict[str, list[dict]]:
    files = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
    arms: dict[str, list[dict]] = {"A": [], "B0": [], "B1": [], "B2": []}
    n_parity = 0

    for f in files:
        text, meta = handle_markdown(f)
        if not text.strip():
            continue
        rel = f"{f.parent.name}/{f.name}"
        verify_parity(text, rel)
        n_parity += 1

        for cid, body, path in tracked_chunker(text, source_path=rel):
            rec = {"id": cid, "source_path": rel, "heading_path": path}
            arms["A"].append({**rec, "text": body})
            arms["B0"].append({**rec, "text": f"{rel}\n\n{body}"})
            arms["B1"].append({
                **rec,
                "text": f"{path}\n\n{body}" if path else body,
            })

        for cid, body, path in section_chunker(text, source_path=rel):
            arms["B2"].append({
                "id": cid, "source_path": rel, "heading_path": path,
                "text": f"{path}\n\n{body}" if path else body,
            })

    print(f"parity verified on {n_parity}/{len(files)} files")
    return arms


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    arms = build()
    for name, chunks in arms.items():
        p = out_dir / f"chunks_{name}.jsonl"
        with p.open("w") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        lens = [len(c["text"]) for c in chunks]
        with_path = sum(1 for c in chunks if c["heading_path"])
        print(
            f"{name:3s} {len(chunks):5d} chunks  "
            f"mean {sum(lens)/len(lens):6.1f} chars  "
            f"with heading path {with_path/len(chunks):5.1%}  -> {p.name}"
        )

    a_ids = [c["id"] for c in arms["A"]]
    for other in ("B0", "B1"):
        assert [c["id"] for c in arms[other]] == a_ids, f"{other} boundaries drifted from A"
    assert [c["text"] for c in arms["A"]] == [
        c["text"].split("\n\n", 1)[1] if c["heading_path"] else c["text"]
        for c in arms["B1"]
    ], "B1 bodies differ from A"
    print("A/B0/B1 boundary + body identity checks passed")
