"""Minimal Gemini client through the Cloudflare AI Gateway, with a system instruction and a JSON schema.

The invoking-gemini skill's `invoke_gemini` has no system-instruction argument, and the drafter here has to
see the same system prompt as the Opus drafter in positive-writer-brief, so this sends the REST body itself.

    generate(system, prompt, schema, model, thinking_level="low") -> (parsed | None, meta)

`meta` holds usageMetadata, finishReason and, on failure, the last error. A failure returns None, never a
partial or empty object.
"""
from __future__ import annotations

import json
import os
import random
import time

import requests

BASE = "https://gateway.ai.cloudflare.com/v1/{acct}/{gw}/google-ai-studio/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}


def generate(system: str, prompt: str, schema: dict, model: str, thinking_level: str | None = "low",
             max_output_tokens: int = 32768, retries: int = 6, timeout: int = 300):
    url = BASE.format(acct=os.environ["CF_ACCOUNT_ID"], gw=os.environ["CF_GATEWAY_ID"], model=model)
    headers = {"Content-Type": "application/json",
               "cf-aig-authorization": "Bearer " + os.environ["CF_API_TOKEN"]}
    cfg = {"responseMimeType": "application/json", "responseJsonSchema": schema,
           "maxOutputTokens": max_output_tokens}
    if thinking_level:
        cfg["thinkingConfig"] = {"thinkingLevel": thinking_level}
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg}
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                j = r.json()
                cand = (j.get("candidates") or [{}])[0]
                finish = cand.get("finishReason")
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                meta = {"usage": j.get("usageMetadata", {}), "finish": finish,
                        "model_version": j.get("modelVersion"), "seconds": round(time.time() - t0, 1),
                        "attempts": attempt + 1}
                try:
                    return json.loads(text), meta
                except json.JSONDecodeError:
                    err = f"unparseable output, finish={finish}, {len(text)} chars"
            elif r.status_code in RETRY_STATUS:
                err = f"HTTP {r.status_code}: {r.text[:300]}"
            else:
                return None, {"error": f"HTTP {r.status_code}: {r.text[:600]}", "attempts": attempt + 1}
        time.sleep(min(60, 2 ** attempt + random.random() * 2))
    return None, {"error": err, "attempts": retries}
