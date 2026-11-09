"""Metered LLM gateway: every model call in every arm goes through here.

There is no ANTHROPIC_API_KEY in this environment, so calls run as headless
`claude -p` children with tools, skills, MCP and settings switched off. Each
call's `modelUsage` (per-model token totals for the run) is priced at list
rates and appended to the run's ledger. The ledger is the only stopping rule
besides the wall clock: once a run has spent its cap, `call` raises.

Two costs are recorded per call:

  cost      as billed: the call's real token counts at list prices, including
            the ~1.4K tokens Claude Code adds to every request and its 1-hour
            cache writes (2x input). This is the PoC's dollar axis.
  cost_api  what the same request would cost on the raw Messages API with the
            harness's per-call overhead removed (measured once per model by
            `calibrate`) and cache writes at the 5-minute rate (1.25x). A
            secondary axis: the overhead falls hardest on many small calls.

Arm A reaches the gateway through `serve_openai`, an OpenAI-compatible
chat-completions endpoint on a Unix socket that the jailed OpenEvolve reaches
through jailside/bridge.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# USD per million tokens, list prices (claude-api skill model table, cached
# 2026-10-06). Haiku 5.5 bills 5x above 100K prompt tokens; prompts here stay
# far below that, but the tier is applied if one ever crosses it.
PRICES = {
    "claude-haiku-5-5": {"inp": 0.10, "out": 0.50},
    "claude-sonnet-5-5": {"inp": 2.00, "out": 10.00},
}
READ_MULT, W5M_MULT, W1H_MULT = 0.10, 1.25, 2.00
HAIKU_LONG = 100_000
# Calls that share a prefix land seconds apart, so the 5-minute cache TTL is the
# right one, and its writes cost 1.25x input against the CLI's default 1-hour 2x.
# Every run since 2026-11-09 14:10Z uses it; a run dir records it in `cache_ttl`.
CACHE_TTL = "5m"

ALIASES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}


class BudgetExceeded(RuntimeError):
    pass


class QuotaGuard(RuntimeError):
    """The subscription's rate-limit window is nearly spent; stop the schedule."""


def price(model: str, u: dict, overhead: dict | None = None, api_equiv: bool = False) -> float:
    p = PRICES[model]
    inp = u.get("inputTokens", 0)
    out = u.get("outputTokens", 0)
    read = u.get("cacheReadInputTokens", 0)
    w5 = u.get("cacheCreationEphemeral5mInputTokens", 0)
    w1 = u.get("cacheCreationEphemeral1hInputTokens", 0)
    # Some responses carry only the combined creation count.
    rest = u.get("cacheCreationInputTokens", 0) - w5 - w1
    if rest > 0:
        w1 += rest
    if api_equiv and overhead:
        # Remove the harness's own per-call tokens: the part that is re-written
        # every call and the part that is read from cache.
        o_new, o_read = overhead.get("new", 0), overhead.get("read", 0)
        cut = min(w1, o_new); w1 -= cut; o_new -= cut
        cut = min(w5, o_new); w5 -= cut; o_new -= cut
        inp = max(0, inp - o_new)
        read = max(0, read - o_read)
    total_in = inp + read + w5 + w1
    mult = 5.0 if (model == "claude-haiku-5-5" and total_in > HAIKU_LONG) else 1.0
    ip, op = p["inp"] * mult / 1e6, p["out"] * mult / 1e6
    w1_mult = W5M_MULT if api_equiv else W1H_MULT
    return inp * ip + out * op + read * ip * READ_MULT + w5 * ip * W5M_MULT + w1 * ip * w1_mult


@dataclass
class Gateway:
    run_dir: Path
    cap: float
    max_parallel: int = 16
    quota_stop: float = 0.90        # five-hour utilisation that pauses a run with Sonnet calls until it resets
    quota_stop_haiku: float = 0.97  # Haiku calls cost ~20x less, so a Haiku-only run keeps going longer
    week_stop: float = 0.45         # seven-day utilisation that aborts the run
    overhead: dict = field(default_factory=dict)
    spent: float = 0.0
    spent_api: float = 0.0
    calls: int = 0
    last_quota: dict = field(default_factory=dict)
    pauses: list = field(default_factory=list)   # [(start, end)] quota waits, excluded from wall clock

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.run_dir / "ledger.jsonl"
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(self.max_parallel)
        self._pause_lock = threading.Lock()
        pf = self.run_dir / "pauses.jsonl"
        if pf.exists():
            for line in pf.read_text().splitlines():
                r = json.loads(line)
                self.pauses.append((r["start"], min(r["until"], time.time())))
        self._cwd = tempfile.mkdtemp(prefix="hsp-cc-")
        self.t0 = time.time()
        (self.run_dir / "cache_ttl").write_text(CACHE_TTL)
        # Resume: re-read an existing ledger so the cap holds across restarts.
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                r = json.loads(line)
                self.spent += r.get("cost", 0.0)
                self.spent_api += r.get("cost_api", 0.0)
                self.calls += 1
        ov = self.run_dir / "overhead.json"
        if ov.exists() and not self.overhead:
            self.overhead = json.loads(ov.read_text())

    # ------------------------------------------------------------------ calls
    def call(self, model: str, system: str, user: str, *, effort: str,
             max_tokens: int = 16000, role: str = "", meta: dict | None = None,
             retries: int = 2, check_budget: bool = True) -> dict:
        model = ALIASES.get(model, model)
        for attempt in range(retries + 1):
            with self._pause_lock:
                pass                  # blocks while another call holds a quota pause
            if model != "claude-haiku-5-5" and (self.last_quota.get("five_hour") or 0) >= self.quota_stop:
                self._wait_for_reset(self.last_quota)
            if check_budget and self.spent >= self.cap:
                raise BudgetExceeded(f"spent {self.spent:.4f} >= cap {self.cap}")
            with self._sem:
                if check_budget and self.spent >= self.cap:
                    raise BudgetExceeded(f"spent {self.spent:.4f} >= cap {self.cap}")
                rec = self._run_cli(model, system, user, effort, max_tokens)
            rec.update(role=role, meta=meta or {}, attempt=attempt)
            self._record(rec)
            if rec["ok"]:
                return rec
            err = (rec.get("error") or "").lower()
            if "rate limit" in err or "usage limit" in err or "429" in err:
                raise QuotaGuard(rec.get("error"))
            time.sleep(3 * (attempt + 1))
        return rec

    def _run_cli(self, model, system, user, effort, max_tokens) -> dict:
        sp = Path(self._cwd) / f"sys-{uuid.uuid4().hex}.txt"
        sp.write_text(system)
        cmd = ["claude", "-p", "--model", model, "--no-session-persistence",
               "--output-format", "stream-json", "--verbose",
               "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--disable-slash-commands", "--exclude-dynamic-system-prompt-sections",
               "--effort", effort, "--system-prompt-file", str(sp)]
        env = dict(os.environ, CLAUDE_CODE_EFFORT_LEVEL=effort,
                   CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(max_tokens),
                   FORCE_PROMPT_CACHING_5M="1", CLAUDE_CODE_PROMPT_CACHE_TTL=CACHE_TTL)
        t_start = time.time()
        try:
            p = subprocess.run(cmd, input=user, capture_output=True, text=True,
                               timeout=900, cwd=self._cwd, env=env)
            out, rc, stderr = p.stdout, p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            out, rc, stderr = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), -9, "timeout"
        finally:
            sp.unlink(missing_ok=True)
        t_end = time.time()
        result, quota = None, None
        for line in out.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "result":
                result = d
            elif d.get("type") == "rate_limit_event":
                quota = d.get("rate_limit_info")
        rec = {"t_start": t_start, "t_end": t_end, "model": model, "effort": effort,
               "ok": False, "text": "", "usage": {}, "cost": 0.0, "cost_api": 0.0,
               "cost_cc": None, "error": None}
        if quota:
            win = quota.get("unifiedWindows") or {}
            rec["quota"] = {k: win.get(k, {}).get("utilization") for k in ("five_hour", "seven_day")}
            rec["quota"]["status"] = quota.get("status")
            rec["quota"]["five_hour_resets"] = (win.get("five_hour") or {}).get("resetsAt")
        if result is None:
            rec["error"] = f"rc={rc} no result; stderr={stderr[-400:]}"
            return rec
        mu = result.get("modelUsage") or {}
        u = mu.get(model) or (next(iter(mu.values())) if mu else {})
        rec["usage"] = {k: u.get(k, 0) for k in (
            "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens",
            "cacheCreationEphemeral5mInputTokens", "cacheCreationEphemeral1hInputTokens",
            "thinkingTokens")}
        if len(mu) > 1:
            rec["error_extra_models"] = list(mu)
        rec["cost"] = sum(price(m, v) for m, v in mu.items() if m in PRICES)
        rec["cost_api"] = sum(price(m, v, self.overhead.get(m), api_equiv=True)
                              for m, v in mu.items() if m in PRICES)
        rec["cost_cc"] = result.get("total_cost_usd")
        rec["num_turns"] = result.get("num_turns")
        rec["stop_reason"] = result.get("stop_reason")
        if result.get("is_error") or result.get("subtype") != "success":
            rec["error"] = str(result.get("result") or result.get("subtype"))[:400]
            return rec
        rec["text"] = result.get("result") or ""
        rec["ok"] = True
        return rec

    def _record(self, rec: dict):
        with self._lock:
            self.spent += rec["cost"]
            self.spent_api += rec["cost_api"]
            self.calls += 1
            rec["spent_after"] = self.spent
            rec["spent_api_after"] = self.spent_api
            row = {k: v for k, v in rec.items() if k != "text"}
            row["text_len"] = len(rec.get("text") or "")
            with self.ledger.open("a") as f:
                f.write(json.dumps(row) + "\n")
            q = rec.get("quota") or {}
            if q:
                self.last_quota = q
        if q and (q.get("seven_day") or 0) >= self.week_stop:
            raise QuotaGuard(f"quota utilisation {q}")
        limit = self.quota_stop_haiku if rec.get("model") == "claude-haiku-5-5" else self.quota_stop
        if q and (q.get("five_hour") or 0) >= limit:
            self._wait_for_reset(q)

    def _wait_for_reset(self, q: dict):
        """Hold every call until the five-hour window resets. The pause is logged
        and subtracted from the run's wall clock; spend is unaffected."""
        with self._pause_lock:
            reset = q.get("five_hour_resets") or (time.time() + 1800)
            now = time.time()
            if now >= reset or (self.pauses and self.pauses[-1][1] >= reset):
                return
            with (self.run_dir / "pauses.jsonl").open("a") as f:
                f.write(json.dumps({"start": now, "until": reset + 30, "quota": q}) + "\n")
            time.sleep(max(0, reset + 30 - now))
            self.pauses.append((now, time.time()))

    def paused_before(self, t: float) -> float:
        """Seconds of quota pause that ended before time t."""
        return sum(min(e, t) - s for s, e in self.pauses if s < t)

    # ------------------------------------------------------------ calibration
    def calibrate(self, models=("claude-haiku-5-5", "claude-sonnet-5-5")):
        """Measure the harness's own per-call tokens with a one-token prompt.

        Two identical calls per model: the second shows which part of the
        overhead is read from cache and which part is written anew each call.
        Calibration calls are charged to the ledger with role='calibrate'.
        """
        out = {}
        for m in models:
            m = ALIASES.get(m, m)
            recs = [self.call(m, ".", "Reply with the single character: .", effort="low",
                              max_tokens=64, role="calibrate", check_budget=False) for _ in range(2)]
            u = recs[1]["usage"]
            out[m] = {"new": u.get("cacheCreationInputTokens", 0) + u.get("inputTokens", 0),
                      "read": u.get("cacheReadInputTokens", 0)}
        self.overhead = out
        (self.run_dir / "overhead.json").write_text(json.dumps(out, indent=1))
        return out


# ---------------------------------------------------------------- Arm A shim
def serve_openai(gw: Gateway, sock_path: str, *, effort: str, model_map: dict | None = None):
    """OpenAI chat-completions over HTTP on a Unix socket, backed by `gw`.

    Sampling parameters (temperature, top_p, seed) are dropped: `claude -p`
    cannot set them, and Sonnet 5.5 rejects non-default sampling values on the
    Messages API anyway.
    """
    import socketserver
    from http.server import BaseHTTPRequestHandler

    model_map = model_map or {}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def address_string(self):
            return "uds"

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._send(404, {"error": {"message": f"no route {self.path}"}})
            msgs = req.get("messages") or []
            system = "\n\n".join(_text(m) for m in msgs if m.get("role") in ("system", "developer"))
            turns = [m for m in msgs if m.get("role") not in ("system", "developer")]
            if len(turns) == 1:
                user = _text(turns[0])
            else:
                user = "\n\n".join(f"[{m.get('role')}]\n{_text(m)}" for m in turns)
            model = model_map.get(req.get("model"), req.get("model"))
            max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 16000)
            try:
                rec = gw.call(model, system or "You are a helpful assistant.", user,
                              effort=effort, max_tokens=max_tokens, role="oe")
            except BudgetExceeded as e:
                return self._send(429, {"error": {"message": str(e), "type": "budget_exceeded"}})
            except QuotaGuard as e:
                return self._send(429, {"error": {"message": str(e), "type": "quota_guard"}})
            if not rec["ok"]:
                return self._send(500, {"error": {"message": rec.get("error") or "call failed"}})
            u = rec["usage"]
            pt = u.get("inputTokens", 0) + u.get("cacheReadInputTokens", 0) + u.get("cacheCreationInputTokens", 0)
            self._send(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                "created": int(time.time()), "model": req.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": rec["text"]}}],
                "usage": {"prompt_tokens": pt, "completion_tokens": u.get("outputTokens", 0),
                          "total_tokens": pt + u.get("outputTokens", 0)},
            })

        def do_GET(self):
            self._send(200, {"object": "list", "data": [{"id": "claude-sonnet-5-5", "object": "model"}]})

    class Srv(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = Srv(sock_path, H)
    os.chmod(sock_path, 0o666)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""
