"""Shared utilities for theory-empirical bridge pipeline.

Extends phase_a/scripts/common.py patterns with asymmetric-corpus paths
and the Gemini 2.5-flash model IDs used in this iteration.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("TE_DATA_DIR", ROOT / "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
S2_API_KEY   = os.environ.get("S2_API_KEY", "")
CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_GATEWAY_ID = os.environ["CF_GATEWAY_ID"]
CF_API_TOKEN  = os.environ["CF_API_TOKEN"]
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

GEMINI_BASE = (
    f"https://gateway.ai.cloudflare.com/v1/{CF_ACCOUNT_ID}/{CF_GATEWAY_ID}"
    "/google-ai-studio/v1beta"
)

# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base: float = 1.5,
    cap: float = 30.0,
    retry_on: tuple = (httpx.HTTPError,),
) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as e:
            last = e
            wait = min(cap, base ** i) + random.uniform(0, 0.5)
            print(f"  retry {i+1}/{attempts} after {wait:.1f}s: {e}", file=sys.stderr)
            time.sleep(wait)
    assert last is not None
    raise last


def chunked(seq: Iterable[Any], n: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"


def s2_headers() -> dict[str, str]:
    return {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


def s2_post(path: str, body: Any, params: dict | None = None, timeout: float = 120.0) -> Any:
    def _call():
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{S2_BASE}{path}", params=params, json=body, headers=s2_headers())
            if r.status_code == 429:
                raise httpx.HTTPError(f"S2 429: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
    return retry(_call, attempts=8, cap=60.0)


def s2_get(path: str, params: dict | None = None, timeout: float = 60.0) -> Any:
    def _call():
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{S2_BASE}{path}", params=params, headers=s2_headers())
            if r.status_code == 429:
                raise httpx.HTTPError(f"S2 429: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
    return retry(_call, attempts=8, cap=60.0)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def gemini_embed_batch(
    texts: list[str],
    *,
    model: str = "gemini-embedding-001",
    dim: int = 768,
    chunk: int = 100,
) -> list[list[float] | None]:
    if not texts:
        return []
    url = f"{GEMINI_BASE}/models/{model}:batchEmbedContents"
    out: list[list[float] | None] = []
    for i in range(0, len(texts), chunk):
        sub = texts[i : i + chunk]
        body = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": dim,
                }
                for t in sub
            ]
        }

        def _call(sub=sub, body=body):
            with httpx.Client(timeout=120.0) as c:
                r = c.post(
                    url,
                    headers={
                        "cf-aig-authorization": f"Bearer {CF_API_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                r.raise_for_status()
                data = r.json()
                embs = data.get("embeddings") or []
                if len(embs) != len(sub):
                    raise httpx.HTTPError(
                        f"batch returned {len(embs)} for {len(sub)} inputs"
                    )
                return [e.get("values") for e in embs]

        try:
            vecs = retry(_call, attempts=5, cap=30.0)
        except Exception as e:
            print(f"  embed batch {i//chunk} failed: {e}", file=sys.stderr)
            vecs = [None] * len(sub)
        out.extend(vecs)
    return out


def gemini_generate(
    prompt: str,
    model: str = "gemini-2.5-flash",
    *,
    json_mode: bool = False,
    max_tokens: int = 2048,
    thinking_budget: int = 0,
) -> str:
    url = f"{GEMINI_BASE}/models/{model}:generateContent"
    cfg: dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": 0.2}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    if thinking_budget >= 0:
        cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": cfg,
    }

    def _call():
        with httpx.Client(timeout=120.0) as c:
            r = c.post(
                url,
                headers={
                    "cf-aig-authorization": f"Bearer {CF_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
            cands = data.get("candidates") or []
            if not cands:
                return ""
            parts = (cands[0].get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts)

    return retry(_call, attempts=5, cap=30.0)


# ---------------------------------------------------------------------------
# JSON checkpointing
# ---------------------------------------------------------------------------

def save_json(name: str, obj: Any) -> Path:
    p = DATA / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(p)
    print(f"  saved {p.relative_to(ROOT)}", file=sys.stderr)
    return p


def load_json(name: str, default: Any = None) -> Any:
    p = DATA / name
    if not p.exists():
        return default
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Unicode → ASCII folding for substring matching
# ---------------------------------------------------------------------------
#
# Diagnosed 2026-05-24 in te_anchor.py's abstract-mention filter (PR #99):
# S2's /paper/{id} response returns Polish/Nordic/Croatian author names with
# their precomposed stroke letters intact (e.g. 'Odrzywołek'), but S2's
# abstract text normalizes those letters to ASCII ('Odrzywolek'). Substring
# match across the two surfaces fails unless both sides are ASCII-folded.
#
# The naive `unicodedata.normalize("NFKD", s).encode("ascii", "ignore")`
# is NOT sufficient: precomposed stroked letters like Polish ł (U+0142),
# Nordic ø (U+00F8), Croatian đ (U+0111), Icelandic þ/ð, German ß, and the
# ligatures æ/œ have no canonical NFKD decomposition. ascii-encode drops
# them silently:
#     'Odrzywołek' → NFKD → 'Odrzywołek' → ascii encode → 'Odrzywoek'
# That missing 'l' breaks substring matching on the surname.
#
# Fix: translate the known-uncomposable stroke letters BEFORE NFKD, then
# fall through to NFKD + ascii-encode for the rest of the diacritic-bearing
# letters (which DO have canonical decompositions: ñ, é, ü, ç, etc.).

_STROKED_LETTER_FOLD = str.maketrans({
    "ł": "l", "Ł": "L",    # Polish
    "ø": "o", "Ø": "O",    # Nordic
    "đ": "d", "Đ": "D",    # Croatian / Vietnamese
    "ð": "d", "Ð": "D",    # Icelandic / Faroese
    "þ": "th", "Þ": "Th",  # Icelandic / Old English
    "æ": "ae", "Æ": "AE",  # Latin / Nordic
    "œ": "oe", "Œ": "OE",  # French
    "ß": "ss",             # German sharp s
    "ı": "i",              # Turkish dotless i
})


def ascii_fold(s: str) -> str:
    """Strip diacritics, fold stroked letters, lowercase. Use this for
    substring matching across mixed-Unicode and ASCII-normalized text
    surfaces (e.g. S2 author names vs. S2 abstract text).
    """
    s = (s or "").translate(_STROKED_LETTER_FOLD)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
