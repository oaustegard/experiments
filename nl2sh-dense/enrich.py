#!/usr/bin/env python3
"""Rewrite the documentation corpus so a 270M model's retriever can find it.

Pleias' Redline — a 321M legal assistant running offline on a Raspberry Pi —
reports that "most of the project's effort went not into the model but into
converting the Red Line Guidebook into a corpus a small model can use reliably",
via semantic chunking on logical units, **context injection** annotating each
chunk with its jurisdiction and legal framework, **entity normalization** of
acronyms and terms, and markdown structure preserved through retrieval
(pleias.ai/blog/local-ai-for-knowledge).

Two of those four this corpus already has. tldr examples are logical units, and
`page_chunks` gives the document hierarchy. The two it lacks are the two that
address the failure this directory has measured from the start: *"recover the
password for backup.zip"* retrieves `funzip` and `bzip2recover` and never
`fcrackzip`, because the page for `fcrackzip` says "dictionary attack" and the
user says "recover the password". The documentation is written in the vocabulary
of the tool, and the query arrives in the vocabulary of the goal.

So this pass asks `gemini-3.5-flash-lite` to add, per page, exactly what is
missing: a normalized one-line description, the goal-level phrasings a person
would type wanting this utility, and a short disambiguation against the tools it
is confusable with. The examples are kept verbatim — they are the model's
exemplar and §6 measured that replacing them costs routing.

**The confound, stated up front.** The primary eval's natural language was
written by `gemini-3.7-flash`. If a Gemini model now writes the corpus's query
vocabulary too, recall can rise because two members of one model family agree on
phrasing rather than because the corpus got better for a human. That is this
repo's own documented trap — an eval whose sides share an author is worth 0.3
accuracy. The control is `--eval nl2bash`: NL2Bash's English was written by human
annotators, so a lift that survives there is not family alignment. Both numbers
are reported and the Gemini-to-Gemini one is never quoted alone.

Refusals are expected on an offensive-security corpus and are not errors:
`gemini_direct.py` already measured this model declining to help with password
cracking. A refused or unparseable page keeps its original text and is counted.

    python3 enrich.py --out data/pages_enriched.jsonl          # ~70 min, resumable
    python3 enrich.py --render --out data/chunks_enriched.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
GATE = experiment("gh-mcp-regex-fit")
sys.path.insert(0, str(RETRIEVAL))
sys.path.insert(0, str(GATE))
import retrieve as R  # noqa: E402
import dense_index as D  # noqa: E402

PROMPT = """You are preparing a shell-documentation corpus for a retrieval system that
serves a very small offline model. Below is the documentation for one command-line
utility.

Utility: {utility}
Documentation:
{body}

Produce JSON with exactly these keys:

"summary": one sentence, at most 20 words, saying what this utility is for in plain
language. Name the goal, not the syntax.

"intents": 4 to 8 short phrases a person would type into a terminal assistant when
they want this utility, WITHOUT knowing its name. Describe the outcome they want.
Use everyday words, not the documentation's words — if the docs say "dictionary
attack", a user might say "recover a forgotten password" or "get into a locked
archive". Do not include the utility's own name in any phrase.

"not_for": at most 2 short clauses naming closely-related tasks this utility does
NOT do, when confusion is likely. Empty list if nothing is confusable.

"category": one or two words for the task family (for example "archive extraction",
"process management", "network scanning").

Output only the JSON object."""


def build_prompt(chunk) -> str:
    return PROMPT.format(utility=chunk.utility, body=chunk.text[:4000])


def parse(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(text)
    except Exception:
        return None
    if not isinstance(d, dict) or "intents" not in d:
        return None
    d["intents"] = [str(x) for x in d.get("intents") or []][:8]
    d["not_for"] = [str(x) for x in d.get("not_for") or []][:2]
    d["summary"] = str(d.get("summary") or "")[:300]
    d["category"] = str(d.get("category") or "")[:60]
    return d


def render(chunk, card: dict | None) -> str:
    """The indexed document. Original text last so the exemplar survives verbatim.

    A page that failed enrichment renders as its original text alone, so the
    enriched corpus is never *worse* than the plain one for that page — the
    comparison isolates what enrichment adds rather than mixing in a loss.
    """
    if not card:
        return chunk.text
    parts = []
    if card.get("summary"):
        parts.append(card["summary"])
    if card.get("category"):
        parts.append(f"Category: {card['category']}")
    if card.get("intents"):
        parts.append("Use when you want to: " + "; ".join(card["intents"]))
    if card.get("not_for"):
        parts.append("Not for: " + "; ".join(card["not_for"]))
    parts.append(chunk.text)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--cards", type=Path, default=HERE / "data" / "cards.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "chunks_enriched.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--render-only", action="store_true",
                    help="skip generation, rebuild the corpus from existing cards")
    a = ap.parse_args()

    pages = D.page_chunks(R.load_chunks(a.chunks))
    if a.limit:
        pages = pages[: a.limit]
    a.cards.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict | None] = {}
    if a.cards.is_file():
        for line in a.cards.open():
            rec = json.loads(line)
            done[rec["id"]] = rec.get("card")
        print(f"resuming: {len(done)} pages already enriched", file=sys.stderr)

    todo = [p for p in pages if p.id not in done]
    if todo and not a.render_only:
        from gemini_client import generate

        # thinking_budget=-1 OMITS thinkingConfig, which is the fast 0-thinking
        # path. Passing 0 is a hard HTTP 400 on flash-lite (METHODS.md).
        def one(page):
            try:
                return page, parse(generate(build_prompt(page), model=a.model,
                                            thinking_budget=-1,
                                            max_output_tokens=1024,
                                            response_json=True))
            except Exception:
                return page, None

        t0 = time.time()
        with a.cards.open("a") as fh, ThreadPoolExecutor(max_workers=2) as ex:
            for n, (page, card) in enumerate(ex.map(one, todo), 1):
                fh.write(json.dumps({"id": page.id, "utility": page.utility,
                                     "card": card}) + "\n")
                fh.flush()
                done[page.id] = card
                if n % 100 == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"  {n}/{len(todo)} ({rate * 60:.0f}/min, "
                          f"{(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left)",
                          flush=True)

    ok = sum(1 for p in pages if done.get(p.id))
    print(f"enriched {ok}/{len(pages)} pages "
          f"({len(pages) - ok} refused or unparseable, kept verbatim)", file=sys.stderr)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for p in pages:
            fh.write(json.dumps({"id": p.id, "utility": p.utility, "kind": p.kind,
                                 "text": render(p, done.get(p.id)),
                                 "runnable": p.runnable}) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
