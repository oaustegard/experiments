#!/usr/bin/env python3
"""Minimal Gemini client through the Cloudflare AI Gateway.

Self-contained rather than importing `phase-a-bridges/scripts/common.py`, because
an experiment should not depend on a sibling's scripts — but the two `METHODS.md`
entries that module paid for are honoured here explicitly:

* **Concurrency starts at 2.** `phase-a-bridges` learned 12 -> 4 -> 2 the hard
  way; `te-bridges` started at 4 anyway and lost 18-20% of its extractions to
  exhausted retries. `MAX_CONCURRENCY = 2` is not a placeholder to tune upward.
* **Thinking budget must be 0 for structured extraction.** Gemini 2.5/3.x
  thinking models consume the whole output budget and return *silently empty*
  responses rather than errors. Rule *authoring* is the opposite case — it is
  reasoning, not extraction — so it passes `thinking_budget=-1` and lets the
  model decide.

Credentials come from `/mnt/project/proxy.env` (CF_ACCOUNT_ID, CF_GATEWAY_ID,
CF_API_TOKEN), auto-sourced into login shells by
`/etc/profile.d/muninn-env.sh`. Harness `bash -c` calls do not source it, so
`load_env()` reads the file directly.
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "gemini_cache"
PROXY_ENV = Path(os.environ.get("MUNINN_PROXY_ENV", "/mnt/project/proxy.env"))

MAX_CONCURRENCY = 2  # see module docstring; do not raise
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")


class CredentialsMissing(RuntimeError):
    pass


class NotAcceptedError(RuntimeError):
    """A 4xx the server will never accept — raised immediately, never retried."""



def load_env() -> dict[str, str]:
    """Read the three CF variables from the environment, else from proxy.env."""
    keys = ("CF_ACCOUNT_ID", "CF_GATEWAY_ID", "CF_API_TOKEN")
    env = {k: os.environ[k] for k in keys if k in os.environ}
    if len(env) < 3 and PROXY_ENV.is_file():
        for line in PROXY_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.replace("export ", "").strip()
            if k in keys:
                env.setdefault(k, v.strip().strip("'\""))
    missing = [k for k in keys if k not in env]
    if missing:
        raise CredentialsMissing(
            f"missing {', '.join(missing)}. Expected in the environment or in "
            f"{PROXY_ENV}, which is {'present' if PROXY_ENV.is_file() else 'ABSENT'}. "
            "In a CCotw session these arrive by mounting /mnt/project."
        )
    return env


def base_url() -> str:
    e = load_env()
    return (f"https://gateway.ai.cloudflare.com/v1/{e['CF_ACCOUNT_ID']}"
            f"/{e['CF_GATEWAY_ID']}/google-ai-studio/v1beta")


def _cache_key(payload: dict, model: str) -> Path:
    import hashlib
    h = hashlib.sha256(json.dumps([model, payload], sort_keys=True).encode()).hexdigest()[:32]
    return CACHE / f"{h}.json"


def generate(prompt: str, *, model: str = DEFAULT_MODEL, thinking_budget: int = 0,
             max_output_tokens: int = 8192, temperature: float = 0.0,
             response_json: bool = False, use_cache: bool = True) -> str:
    """One generate call. Cached on disk so a re-run costs nothing."""
    cfg: dict = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    if thinking_budget >= 0:
        cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if response_json:
        cfg["responseMimeType"] = "application/json"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}

    ck = _cache_key(payload, model)
    if use_cache and ck.is_file():
        return json.loads(ck.read_text())["text"]

    env = load_env()
    url = f"{base_url()}/models/{model}:generateContent"
    headers = {"cf-aig-authorization": f"Bearer {env['CF_API_TOKEN']}",
               "Content-Type": "application/json"}

    last = None
    for attempt in range(5):
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=180.0)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            if 400 <= r.status_code < 500:
                # A 4xx that is not 429 is a request the server will never accept:
                # a bad model name, or a config that model rejects. Retrying it five
                # times with backoff turns an instant answer into minutes of silence.
                # Measured 2026-08-19: `thinkingBudget: 0` on gemini-3.5-flash-lite
                # is a hard 400, and probing four model names took ~6 minutes because
                # every one of them was retried.
                raise NotAcceptedError(
                    f"HTTP {r.status_code} for model {model}: {r.text[:300]}")
            r.raise_for_status()
            data = r.json()
            cands = data.get("candidates") or []
            parts = (cands[0].get("content", {}).get("parts") or []) if cands else []
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                # The documented silent-empty failure. Surface it as an error.
                raise httpx.HTTPError(
                    f"empty completion (finishReason="
                    f"{cands[0].get('finishReason') if cands else 'none'}); "
                    "if thinking is enabled this is the thinking-budget trap")
            ck.parent.mkdir(parents=True, exist_ok=True)
            ck.write_text(json.dumps({"model": model, "text": text}))
            return text
        except NotAcceptedError:
            raise
        except Exception as e:  # noqa: BLE001 - retried below
            last = e
            time.sleep(min(30.0, 2.0 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"gemini generate failed after 5 attempts: {last}")


def generate_many(prompts: list[str], **kw) -> list[str | None]:
    """Batch at MAX_CONCURRENCY. A failed prompt yields None rather than aborting."""
    def one(p):
        try:
            return generate(p, **kw)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        return list(ex.map(one, prompts))


def available() -> tuple[bool, str]:
    try:
        load_env()
        return True, "credentials present"
    except CredentialsMissing as e:
        return False, str(e)


if __name__ == "__main__":
    ok, why = available()
    print(f"gateway credentials: {'OK' if ok else 'UNAVAILABLE'} — {why}")
    print(f"model: {DEFAULT_MODEL}   max concurrency: {MAX_CONCURRENCY}")
    if ok:
        print(generate("Reply with exactly: pong", max_output_tokens=16)[:100])
