"""Extract chunks from muninn.austegard.com's HTML corpus.

Walks blog/, perch/, scratch/ under the site root; for each .html that
is not a template or index page:

  - Parse with BeautifulSoup.
  - Pull post-level metadata from the document head: <title>,
    <meta name="description">, <meta property="og:url">,
    <meta property="article:published_time">.
  - Strip <nav>/<footer>/<header>/<script>/<style>.
  - Locate the body container in this preference order:
      <article> → <main> → <body>.
  - Convert to plain text (separator="\n", strip=True).
  - Pass through ``remax_kb.pack.default_chunker`` with target_chars=500.
  - Reject chunks shorter than 60 chars.
  - Decorate each chunk's meta with post-level fields.

Returns a list of ``remax_kb.pack.Chunk`` ready to feed ``pack(...)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from remax_kb.pack import Chunk, default_chunker


SECTIONS = ("blog", "perch", "scratch")
SKIP_NAMES = {"_template.html", "index.html"}
MIN_CHUNK_CHARS = 60
TARGET_CHARS = 500


@dataclass(frozen=True)
class PostMeta:
    title: str
    description: str
    url: str
    date: str
    source_path: str  # e.g. "blog/foo.html", relative to site root


def _meta_content(soup: BeautifulSoup, attrs: dict) -> str:
    tag = soup.find("meta", attrs=attrs)
    if tag is None:
        return ""
    return (tag.get("content") or "").strip()


def parse_post(html_path: Path, site_root: Path) -> tuple[PostMeta, str] | None:
    """Parse one HTML file. Returns (meta, body_text) or None if empty."""
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    description = _meta_content(soup, {"name": "description"})
    url = _meta_content(soup, {"property": "og:url"})
    if not url:
        # Most posts don't ship an og:url meta; synthesize from path so
        # retrieved chunks are still linkable.
        url = "https://muninn.austegard.com/" + html_path.relative_to(site_root).as_posix()
    date = _meta_content(soup, {"name": "article:published_time"}) or _meta_content(
        soup, {"property": "article:published_time"}
    )

    # Pick the most specific body container available.
    container = soup.find("article") or soup.find("main") or soup.find("body")
    if container is None:
        return None

    # Strip non-content elements in place.
    for tag in container.find_all(["nav", "footer", "header", "script", "style"]):
        tag.decompose()

    body = container.get_text(separator="\n", strip=True)
    if not body.strip():
        return None

    rel = html_path.relative_to(site_root).as_posix()
    meta = PostMeta(
        title=title,
        description=description,
        url=url,
        date=date,
        source_path=rel,
    )
    return meta, body


def extract_chunks(site_root: str | Path) -> list[Chunk]:
    root = Path(site_root)
    if not root.exists():
        raise FileNotFoundError(root)

    posts: list[tuple[PostMeta, str]] = []
    for section in SECTIONS:
        section_dir = root / section
        if not section_dir.is_dir():
            continue
        for html_path in sorted(section_dir.glob("*.html")):
            if html_path.name in SKIP_NAMES:
                continue
            parsed = parse_post(html_path, root)
            if parsed is None:
                continue
            posts.append(parsed)

    all_chunks: list[Chunk] = []
    for meta, body in posts:
        raw_chunks = default_chunker(
            body, source_path=meta.source_path, target_chars=TARGET_CHARS
        )
        kept = 0
        for c in raw_chunks:
            if len(c.text) < MIN_CHUNK_CHARS:
                continue
            merged = {
                **c.meta,
                "title": meta.title,
                "description": meta.description,
                "url": meta.url,
                "date": meta.date,
                "source_path": meta.source_path,
            }
            all_chunks.append(Chunk(id=c.id, text=c.text, meta=merged))
            kept += 1
        # quiet — no per-post log unless debugging
    return all_chunks


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("site_root", help="Path to muninn.austegard.com checkout")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    chunks = extract_chunks(args.site_root)
    print(f"total chunks: {len(chunks)}", file=sys.stderr)
    print(f"unique source files: {len({c.meta['source_path'] for c in chunks})}", file=sys.stderr)
    if args.limit:
        for c in chunks[: args.limit]:
            print(f"--- {c.id}")
            print(f"  title: {c.meta.get('title','')}")
            print(f"  url:   {c.meta.get('url','')}")
            print(f"  date:  {c.meta.get('date','')}")
            print(f"  text:  {c.text[:160]}...")
