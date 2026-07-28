"""Stage 4 — Body extract + Gemini re-rank.

For each unique paper in the candidate_pairs top-100:
  - Fetch full body text from arXiv (HTML preferred, else abs+source)
  - Skip the first 2000 chars (abstract-region noise per phase-0)
  - Take chars [2000:8000] as the body-only snippet
  - Embed with gemini-embedding-001 at 768 dims via CF AI Gateway

Then recompute distances for the 100 candidate pairs in Gemini-body
space and re-rank, keeping top 20.

arXiv HTML format: https://arxiv.org/html/{arxiv_id}v1 — works for most
papers from 2024+, falls back to /abs/ page text (which has the abstract
+ some metadata, not ideal but better than nothing).

Output:
  data/body_embeddings.json — {arxiv_id: vector}
  data/reranked_pairs.json  — top 20 pairs after body re-rank
"""
from __future__ import annotations

import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gemini_embed, load_json, save_json  # noqa: E402

BODY_START, BODY_END = 2000, 8000
TOP_K_PAIRS = 20
ARXIV_FETCH_SLEEP = 1.0  # be polite


class _TextExtractor(HTMLParser):
    """Tag-stripper that keeps text from <p>, <span>, etc, drops <script>/<style>."""
    SKIP = {"script", "style", "math", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: ANN001
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        # Add whitespace at block tag boundaries to avoid running words together
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "br", "section"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


def fetch_body(arxiv_id: str) -> str:
    """Fetch best-effort body text from arXiv. Tries html/, falls back to abs/."""
    headers = {"User-Agent": "phase-a-mvp/0.1 (research@local)"}
    # 1) arXiv HTML
    for url in (
        f"https://arxiv.org/html/{arxiv_id}v1",
        f"https://arxiv.org/html/{arxiv_id}",
    ):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as c:
                r = c.get(url)
                if r.status_code == 200 and "<html" in r.text.lower():
                    p = _TextExtractor()
                    p.feed(r.text)
                    body = p.text()
                    if len(body) >= 1000:
                        return body
        except httpx.HTTPError:
            pass
    # 2) Fallback: /abs/ page (abstract + metadata only; better than nothing)
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as c:
            r = c.get(f"https://arxiv.org/abs/{arxiv_id}")
            if r.status_code == 200:
                p = _TextExtractor()
                p.feed(r.text)
                return p.text()
    except httpx.HTTPError:
        pass
    return ""


def body_window(text: str) -> str:
    return text[BODY_START:BODY_END] if len(text) >= BODY_START + 500 else text[:BODY_END]


def main() -> None:
    cps = load_json("candidate_pairs.json")
    if not cps:
        print("ERROR: run stage3 first.", file=sys.stderr)
        sys.exit(1)
    pairs = cps["pairs"]
    print(f"Phase A stage 4: body re-rank over {len(pairs)} candidate pairs")

    # Unique arxiv_ids in candidate set
    uniq_ids = sorted({p["a"]["arxiv_id"] for p in pairs} | {p["b"]["arxiv_id"] for p in pairs})
    print(f"  unique papers to fetch + embed: {len(uniq_ids)}")

    # Resume cache
    embeddings = load_json("body_embeddings.json", default={}) or {}
    body_meta = load_json("body_meta.json", default={}) or {}

    pending = [a for a in uniq_ids if a not in embeddings]
    print(f"  cached: {len(uniq_ids) - len(pending)}  to fetch: {len(pending)}")

    for n_done, arxiv_id in enumerate(pending, start=1):
        t0 = time.time()
        body = fetch_body(arxiv_id)
        body_text = body_window(body)
        if not body_text or len(body_text) < 200:
            print(f"  [{n_done}/{len(pending)}] {arxiv_id}: body too short ({len(body_text)} chars), skipping")
            body_meta[arxiv_id] = {"body_len": len(body), "snippet_len": len(body_text), "embedded": False}
            embeddings[arxiv_id] = None  # marker
            continue
        try:
            v = gemini_embed(body_text)
        except Exception as e:
            print(f"  [{n_done}/{len(pending)}] {arxiv_id}: embed err: {e}")
            embeddings[arxiv_id] = None
            body_meta[arxiv_id] = {"body_len": len(body), "snippet_len": len(body_text), "embedded": False, "err": str(e)}
            continue
        embeddings[arxiv_id] = v
        body_meta[arxiv_id] = {"body_len": len(body), "snippet_len": len(body_text), "embedded": True}
        print(f"  [{n_done}/{len(pending)}] {arxiv_id}: body={len(body)} snip={len(body_text)} emb=ok  ({time.time() - t0:.1f}s)")
        # Checkpoint every 10 papers
        if n_done % 10 == 0:
            save_json("body_embeddings.json", embeddings)
            save_json("body_meta.json", body_meta)
        time.sleep(ARXIV_FETCH_SLEEP)

    # Final save
    save_json("body_embeddings.json", embeddings)
    save_json("body_meta.json", body_meta)

    # --- Re-rank pairs ----------------------------------------------------
    def cosine_dist(u: list[float], v: list[float]) -> float:
        a = np.asarray(u, dtype=np.float32)
        b = np.asarray(v, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 1.0
        return float(1.0 - (a @ b) / denom)

    rescored: list[dict] = []
    for p in pairs:
        a, b = p["a"]["arxiv_id"], p["b"]["arxiv_id"]
        va, vb = embeddings.get(a), embeddings.get(b)
        if va is None or vb is None:
            continue
        rescored.append({
            **p,
            "gemini_body_cosine_dist": round(cosine_dist(va, vb), 6),
        })
    rescored.sort(key=lambda x: x["gemini_body_cosine_dist"])
    top = rescored[:TOP_K_PAIRS]

    save_json("reranked_pairs.json", {
        "n_input": len(pairs),
        "n_with_embeddings": len(rescored),
        "n_output": len(top),
        "pairs": top,
    })

    print(f"\nStage 4 complete:")
    print(f"  pairs with both embeddings: {len(rescored)}/{len(pairs)}")
    print(f"  top {TOP_K_PAIRS} written to reranked_pairs.json")
    print(f"\nTop 5 reranked pairs:")
    for i, p in enumerate(top[:5]):
        print(f"  {i+1}. cos={p['gemini_body_cosine_dist']:.4f}  ham={p['distance_norm']:.4f}  "
              f"{p['a']['arxiv_id']} ↔ {p['b']['arxiv_id']}  [{p['kind']}]")


if __name__ == "__main__":
    main()
