"""Stage 4+5 — arXiv body fetch + asymmetric slot extraction.

Fetches full-body text for every unique paper in te_candidates.json,
then extracts two different slot schemas:

  Empirical paper slots:
    phenomenon, regime, mechanism_unknown

  Theory paper slots:
    theorem_claim, regime, mechanism_provided

Both schemas normalize to "X exhibits Y at regime Z" shape so that
embedding distance across them collapses the vocabulary gap.

Output files:
  data/te_bodies.json      — {arxiv_id: text}
  data/te_extractions.json — {arxiv_id: {pool: "empirical"|"theory", slots: {...}}}

Resumable: te_bodies.json and te_extractions.json are updated
incrementally with every checkpoint save.

Usage:
  python te_extract.py [--parallelism 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
import re

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import gemini_generate, load_json, save_json  # noqa: E402

ARXIV_INTERVAL = 1.2  # polite inter-fetch delay
BODY_START, BODY_END = 2000, 8000
CHECKPOINT_EVERY = 25


# ---------------------------------------------------------------------------
# arXiv body fetch (identical to phase_a/stage4_rerank.py pattern)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "math", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag) -> None:
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "br", "section"}:
            self.parts.append(" ")

    def handle_data(self, data) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


def fetch_body(arxiv_id: str) -> str:
    headers = {"User-Agent": "te-bridge-mvp/0.1 (research)"}
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


# ---------------------------------------------------------------------------
# Slot extraction prompts
# ---------------------------------------------------------------------------

EMPIRICAL_EXTRACT_PROMPT = """You are extracting structured fields from an empirical ML/CS research paper.

Title: {title}
Body snippet:
\"\"\"
{body}
\"\"\"

Return a JSON object with exactly these three keys. Focus on the ABSTRACT STRUCTURE of the claim, stripping experimental setup and implementation details. The extracted text will be embedded to find matching theoretical results.

- phenomenon: One sentence stating what empirical observation/behavior the paper reports. Use "X exhibits Y at regime Z" structure. Focus on the abstract claim, not the experimental apparatus.
- regime: One sentence stating where this observation holds: what scale, what model class, what distribution, what conditions.
- mechanism_unknown: One sentence stating what the authors admit they cannot explain, or explicitly call out as open ("we leave the theoretical understanding..."). If no such admission, write "not stated".

Output ONLY the JSON object."""

THEORY_EXTRACT_PROMPT = """You are extracting structured fields from a theoretical mathematics paper.

Title: {title}
Body snippet:
\"\"\"
{body}
\"\"\"

Return a JSON object with exactly these three keys. Focus on the ABSTRACT STRUCTURE of the result, stripping proof technique details. The extracted text will be embedded to find matching empirical observations.

- theorem_claim: One sentence stating what the paper proves. Use "X exhibits Y at regime Z" structure. Focus on the abstract claim, not the proof method.
- regime: One sentence stating where this result applies: hypotheses on the domain, dimensionality, distribution assumptions, parameter ranges.
- mechanism_provided: One sentence stating the key technique or mathematical object that makes the proof work.

Output ONLY the JSON object."""


def extract_slots(arxiv_id: str, pool: str, title: str, body: str) -> dict | None:
    if not body or len(body) < 200:
        return None
    snippet = body_window(body)
    prompt = (EMPIRICAL_EXTRACT_PROMPT if pool == "empirical" else THEORY_EXTRACT_PROMPT).format(
        title=title, body=snippet
    )
    raw = gemini_generate(prompt, model="gemini-2.5-flash", json_mode=True,
                          max_tokens=600, thinking_budget=0)
    s = raw.strip().lstrip("`").lstrip("json").lstrip()
    s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallelism", type=int, default=4)
    args = ap.parse_args()

    candidates = load_json("te_candidates.json")
    if not candidates:
        print("ERROR: run te_scan.py first", file=sys.stderr)
        sys.exit(1)
    pairs = candidates["pairs"]

    emp_meta = {m["arxiv_id"]: m for m in (load_json("empirical_meta.json") or [])}
    th_meta  = {m["arxiv_id"]: m for m in (load_json("theory_meta.json")  or [])}

    # All unique papers with their pool label
    paper_pool: dict[str, str] = {}
    for p in pairs:
        paper_pool[p["emp_arxiv"]] = "empirical"
        paper_pool[p["th_arxiv"]]  = "theory"

    bodies = dict(load_json("te_bodies.json", default={}) or {})
    extractions = dict(load_json("te_extractions.json", default={}) or {})

    # Stage 4: fetch bodies
    need_bodies = [aid for aid in paper_pool if aid not in bodies]
    print(f"Fetching {len(need_bodies)} bodies ({len(bodies)} cached)…")
    for idx, arxiv_id in enumerate(need_bodies, 1):
        text = fetch_body(arxiv_id)
        bodies[arxiv_id] = text
        ok = len(text) >= 500
        print(f"  [{idx}/{len(need_bodies)}] {arxiv_id}: {'ok' if ok else 'short'} ({len(text)} chars)")
        if idx % CHECKPOINT_EVERY == 0:
            save_json("te_bodies.json", bodies)
        time.sleep(ARXIV_INTERVAL)
    save_json("te_bodies.json", bodies)

    # Stage 5: slot extraction (concurrent)
    need_extract = [
        (aid, pool) for aid, pool in paper_pool.items()
        if aid not in extractions or extractions[aid] is None
    ]
    print(f"\nExtracting slots for {len(need_extract)} papers (parallelism={args.parallelism})…")

    def _extract(aid: str, pool: str) -> tuple[str, dict | None]:
        meta = (emp_meta if pool == "empirical" else th_meta).get(aid, {})
        title = meta.get("title", "")
        body = bodies.get(aid, "")
        slots = extract_slots(aid, pool, title, body)
        return aid, slots

    done = 0
    with ThreadPoolExecutor(max_workers=args.parallelism) as ex:
        futs = {ex.submit(_extract, aid, pool): (aid, pool) for aid, pool in need_extract}
        for fut in as_completed(futs):
            aid, pool = futs[fut]
            try:
                _, slots = fut.result()
            except Exception as e:
                print(f"  extract {aid}: error {e}", file=sys.stderr)
                slots = None
            extractions[aid] = {"pool": pool, "slots": slots}
            done += 1
            ok = slots is not None
            print(f"  [{done}/{len(need_extract)}] {aid} ({pool}): {'ok' if ok else 'fail'}")
            if done % CHECKPOINT_EVERY == 0:
                save_json("te_extractions.json", extractions)

    save_json("te_extractions.json", extractions)
    ok_count = sum(1 for v in extractions.values() if v and v.get("slots"))
    print(f"\nDone: {ok_count}/{len(extractions)} successful extractions")


if __name__ == "__main__":
    main()
