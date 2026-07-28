"""Shared utilities for phase A pipeline: paths, creds, retries, HTTP."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import httpx

ROOT = Path(__file__).resolve().parent.parent
# Data directory is configurable per-run so multiple corpora can coexist
# (run 1 -> data/, run 2 -> run2/, etc.) without overwriting each other.
DATA = Path(os.environ.get("PHASE_A_DATA_DIR", ROOT / "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Credentials — adapted from issue spec.
# The issue references /mnt/project/*.env (a claude.ai desktop convention);
# in CCotw these come straight from the process env.
# ---------------------------------------------------------------------------

S2_API_KEY = os.environ.get("S2_API_KEY", "")  # may be empty -> unauthenticated
CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_GATEWAY_ID = os.environ["CF_GATEWAY_ID"]
CF_API_TOKEN = os.environ["CF_API_TOKEN"]
GH_TOKEN = os.environ.get("GH_TOKEN", "")

GEMINI_BASE = (
    f"https://gateway.ai.cloudflare.com/v1/{CF_ACCOUNT_ID}/{CF_GATEWAY_ID}/google-ai-studio/v1beta"
)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base: float = 1.5,
    cap: float = 30.0,
    retry_on: tuple = (httpx.HTTPError,),
    accept_status: tuple = (200,),
) -> T:
    """Exponential backoff with jitter. Returns the function's value or raises."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as e:
            last = e
            wait = min(cap, base ** i) + random.uniform(0, 0.5)
            print(f"  retry {i + 1}/{attempts} after {wait:.1f}s: {e}", file=sys.stderr)
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
# S2 client
# ---------------------------------------------------------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"


def s2_headers() -> dict[str, str]:
    return {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


def s2_get(path: str, params: dict | None = None, timeout: float = 60.0) -> Any:
    """GET against S2. Handles 429 with backoff."""
    def _call():
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{S2_BASE}{path}", params=params, headers=s2_headers())
            if r.status_code == 429:
                raise httpx.HTTPError(f"S2 429 rate-limited: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
    return retry(_call, attempts=8, cap=60.0)


def s2_post(path: str, body: Any, params: dict | None = None, timeout: float = 120.0) -> Any:
    def _call():
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{S2_BASE}{path}", params=params, json=body, headers=s2_headers())
            if r.status_code == 429:
                raise httpx.HTTPError(f"S2 429 rate-limited: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
    return retry(_call, attempts=8, cap=60.0)


# ---------------------------------------------------------------------------
# Gemini client (via Cloudflare AI Gateway)
# ---------------------------------------------------------------------------

def gemini_embed(text: str, model: str = "gemini-embedding-001", dim: int = 768) -> list[float]:
    """Single-text embedding via CF AI Gateway. Returns vector of length `dim`.

    Prefer gemini_embed_batch() for >1 text — one round trip per call is
    a phase-A wart; batching is ~100x faster.
    """
    return gemini_embed_batch([text], model=model, dim=dim)[0]


def gemini_embed_batch(
    texts: list[str], *, model: str = "gemini-embedding-001", dim: int = 768, chunk: int = 100,
) -> list[list[float] | None]:
    """Batch embedding via :batchEmbedContents. Up to 100 texts per call.

    Returns vectors in input order. None for any text that fails (we don't
    raise mid-batch — a phase-B run can have stragglers without aborting).
    """
    if not texts:
        return []
    url = f"{GEMINI_BASE}/models/{model}:batchEmbedContents"
    out: list[list[float] | None] = []
    for i in range(0, len(texts), chunk):
        sub = texts[i:i + chunk]
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
                    raise httpx.HTTPError(f"batch returned {len(embs)} embeddings for {len(sub)} inputs")
                return [e.get("values") for e in embs]

        try:
            vecs = retry(_call, attempts=5, cap=30.0)
        except Exception as e:
            print(f"  batch {i // chunk} failed entirely: {e}", file=sys.stderr)
            vecs = [None] * len(sub)
        out.extend(vecs)
    return out


def gemini_generate(
    prompt: str,
    model: str = "gemini-3.5-flash",
    *,
    json_mode: bool = False,
    max_tokens: int = 2048,
    thinking_budget: int = 0,
) -> str:
    """Single-turn generate via CF AI Gateway.

    gemini-3.5-flash (and 2.5) think by default. With small max_tokens
    budgets that swallows the whole budget and leaves zero content tokens,
    so we explicitly disable thinking (budget=0) for fast extraction
    calls. Set thinking_budget=-1 to let the model decide (deep reasoning
    calls).
    """
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
