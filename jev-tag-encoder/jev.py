"""Jev 256-tag encoder: one call per text, all 256 nouls, cached per (set, variant).

Transport: Cloudflare Workers AI `typesafe/jev` through the AI Gateway (the TypeSafe
key is stored gateway-side; CF_ACCOUNT_ID + CF_API_TOKEN + CF_GATEWAY_ID in env), or
TypeSafe directly with TYPESAFE_API_KEY. Gateway caching is skipped on every call so
latency and the determinism check measure the model, not the cache.

Cache: cache/<set>__<variant>.jsonl, one line per text, appended as each call lands
(the checkpoint); `to_parquet` writes vectors/<set>__<variant>.parquet keyed by id.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
VECTORS = HERE / "vectors"
TAGS = [t for line in (HERE / "tags.txt").read_text().splitlines() for t in line.split("|")]
assert len(TAGS) == 256

# variant -> (state key, question template)
VARIANTS = {
    "about": ("document", "The document is about {}."),
    "mentions": ("document", "The document mentions {}."),
    "substantially": ("document", "The document is substantially about {}."),
    "query": ("query", "The query asks about {}."),
}


# The AI Gateway is configured at 50 requests per 60 s, fixed window (read from the gateway config
# 2026-09-23). Pace request starts under it instead of backing off exponentially into it.
RPM = float(os.environ.get("JEV_RPM", 48))
_pace_lock = threading.Lock()
_next_start = [0.0]


def _pace() -> None:
    with _pace_lock:
        now = time.time()
        start = max(now, _next_start[0])
        _next_start[0] = start + 60.0 / RPM
    time.sleep(max(0.0, start - now))


class Blocked(RuntimeError):
    """TypeSafe's edge WAF rejected the request bytes (arrives as HTTP 402 via the gateway)."""


def questions(variant: str) -> dict:
    tmpl = VARIANTS[variant][1]
    return {f"t{i:03d}": {"type": "noul", "instructions": tmpl.format(t)} for i, t in enumerate(TAGS)}


def call(text: str, variant: str = "about", timeout: int = 120) -> dict:
    """One Jev call. Returns {"p": [256 floats], "in_tok", "out_tok", "latency_s", "model", "cache"}."""
    state = {VARIANTS[variant][0]: text}
    qs = questions(variant)
    if os.environ.get("CF_API_TOKEN") and os.environ.get("CF_ACCOUNT_ID"):
        url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}/ai/run"
        body = {"model": "typesafe/jev", "input": {"state": state, "questions": qs}}
        headers = {"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}", "Content-Type": "application/json",
                   "cf-aig-skip-cache": "true"}
        if os.environ.get("CF_GATEWAY_ID"):
            headers["cf-aig-gateway-id"] = os.environ["CF_GATEWAY_ID"]
        unwrap = lambda r: r["result"]["result"]  # noqa: E731
    elif os.environ.get("TYPESAFE_API_KEY"):
        url = "https://api.typesafe.ai/v1/systemone"
        body = {"model": "jev-latest", "state": state, "questions": qs}
        headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"}
        unwrap = lambda r: r  # noqa: E731
    else:
        raise RuntimeError("no Jev transport: set CF_ACCOUNT_ID+CF_API_TOKEN(+CF_GATEWAY_ID) or TYPESAFE_API_KEY")
    data = json.dumps(body).encode()
    last = None
    for attempt in range(12):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        _pace()
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = json.load(r)
                cache = r.headers.get("cf-aig-cache-status")
            dt = time.time() - t0
            res = unwrap(raw)
            ans = res["answers"]
            p = [float(ans[f"t{i:03d}"]["noul"]) for i in range(256)]
            u = res.get("usage", {})
            return {"p": p, "in_tok": u.get("input_tokens"), "out_tok": u.get("output_tokens"),
                    "latency_s": round(dt, 3), "model": res.get("model"), "cache": cache}
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")
            if "you have been blocked" in txt or "Attention Required" in txt:
                raise Blocked(f"HTTP {e.code}: WAF block") from None
            last = f"HTTP {e.code}: {txt[:300]}"
            if 400 <= e.code < 500 and e.code not in (408, 409, 429):
                raise RuntimeError(last) from None
            # gateway rate limit (code 2003): the fixed window resets within 60 s
            wait = float(e.headers.get("Retry-After") or 0) or (10 if e.code == 429 else min(60, 2 * 2 ** attempt))
            time.sleep(wait)
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError, ConnectionError) as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 1.5 * 2 ** attempt))
    raise RuntimeError(f"Jev call failed after retries: {last}")


def cache_path(set_name: str, variant: str) -> Path:
    return CACHE / f"{set_name}__{variant}.jsonl"


def load(set_name: str, variant: str) -> dict[str, dict]:
    """id -> record (records with "error" are included; callers filter on "p")."""
    path = cache_path(set_name, variant)
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def encode(set_name: str, variant: str, items: list[tuple[str, str]], concurrency: int = 2,
           log_every: int = 50) -> dict[str, dict]:
    """Encode (id, text) items not already cached. Appends each result as it lands."""
    CACHE.mkdir(exist_ok=True)
    done = load(set_name, variant)
    todo = [(i, t) for i, t in items if i not in done or "p" not in done[i] and done[i].get("error") != "blocked"]
    lock = threading.Lock()
    path = cache_path(set_name, variant)
    n = [0]
    t0 = time.time()

    def one(it):
        i, t = it
        try:
            rec = {"id": i, **call(t, variant)}
        except Blocked:
            rec = {"id": i, "error": "blocked"}
        except RuntimeError as e:
            rec = {"id": i, "error": str(e)[:300]}
        with lock:
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n[0] += 1
            if n[0] % log_every == 0 or n[0] == len(todo):
                print(f"[{set_name}/{variant}] {n[0]}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(one, todo))
    return load(set_name, variant)


def matrix(set_name: str, variant: str, ids: list[str]) -> np.ndarray:
    """(len(ids), 256) float32; rows for missing/failed ids are NaN."""
    recs = load(set_name, variant)
    out = np.full((len(ids), 256), np.nan, dtype=np.float32)
    for k, i in enumerate(ids):
        r = recs.get(i)
        if r and "p" in r:
            out[k] = r["p"]
    return out


def to_parquet(set_name: str, variant: str) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    recs = [r for r in load(set_name, variant).values() if "p" in r]
    VECTORS.mkdir(exist_ok=True)
    tbl = pa.table({
        "id": [r["id"] for r in recs],
        "p": pa.array([np.asarray(r["p"], dtype=np.float32) for r in recs], type=pa.list_(pa.float32(), 256)),
        "in_tok": [r["in_tok"] for r in recs],
        "out_tok": [r["out_tok"] for r in recs],
        "latency_s": [r["latency_s"] for r in recs],
        "model": [r["model"] for r in recs],
    })
    out = VECTORS / f"{set_name}__{variant}.parquet"
    pq.write_table(tbl, out, compression="zstd")
    return out
