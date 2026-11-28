#!/usr/bin/env python3
"""inkling-batch: a stdio MCP server that runs Inkling 2 Mini on erisinky's lane-batched engine, so Claude Code can
fan independent work out across many lanes of one GPU and get hundreds of tokens per second in total.

What it is and is not (measured 2026-11-27 on an M5 Max, see experiments/inkling-fact-forwarding/ENGINE.md):
  * Aggregate decode: about 120 tok/s at 4 lanes, 200 at 8, 340 at 16, 550 at 32, 830-910 at 64, against about
    105 tok/s for the single-stream engine behind `inkling-fast`. Each lane gets slower (24 tok/s at 8 lanes,
    13-14 at 64), so this raises THROUGHPUT for independent items; it does not make one answer arrive sooner.
  * Prefill does not parallelise (about 3.5-4k tok/s however the text is split), so a batch of long prompts is
    prefill-bound. Lanes pay for decode-heavy work and for many samples of one prompt (one shared prefill).
  * A lane's output is not bit-identical to the same prompt run alone, and depends on what runs beside it.

Each call starts the engine, runs one batch and exits, so no GPU memory is held between calls (model load takes
about 0.1 s once the file is in the page cache, about 6 s cold). Calls are served one at a time.

Standard library only. The engine is a child process: no network port, nothing leaves the machine. The tools take
text in and give text out; the model gets no file, shell or network access.

Tools:
  batch_map        N independent prompts, one lane each (continuous batching if N > lanes)
  batch_samples    N samples of ONE prompt from one shared prefill (self-consistency, vote, pass@n), optional answer tally
  batch_split_ask  EXPERIMENTAL divide-and-conquer: split a long text into shards, extract facts per shard in lanes,
                   then answer from the pooled facts with a vote. Accuracy benefit is NOT established.
  batch_status     engine, model, power state and GPU contention

Needs the engine built from upstream erisinky 4be5bed with erisinky-batch-json.patch (next to this file; adds `batch-json`).

Environment (all optional):
  INKLING_WORKSPACE          root for `file` inputs and the default paths (default: cwd)
  INKLING_BATCH_BIN          engine binary (default <workspace>/tools/erisinky-batch/target/release/inkling)
  INKLING_BATCH_MODEL        GGUF (default LM Studio's Inkling-2-Mini-Q4_K_M.gguf)
  INKLING_BATCH_CTX_CAP      largest context a call may use, in tokens (default 65536)
  INKLING_BATCH_MAX_LANES    most lanes in one call (default 64)
  INKLING_BATCH_MAX_MEM_GB   GPU working-memory budget for lanes in GiB (default 48; the engine also caps itself)
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any

# Path components that look like credentials, matched on names not contents. Same pattern as inkling_mcp.py
# (copied so this file stands alone).
_SECRET_NAME = re.compile(
    r"^\.env|\.pem$|\.key$|\.p12$|\.pfx$|\.keychain|^id_(rsa|ed25519|ecdsa|dsa)|"
    r"^\.(aws|ssh|kube|netrc|npmrc|pgpass|git-credentials)$|credential|secret",
    re.IGNORECASE,
)

SERVER_NAME = "inkling-batch"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

DEFAULT_MODEL = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf")
CTX_CAP = int(os.environ.get("INKLING_BATCH_CTX_CAP", "65536"))
MAX_LANES = int(os.environ.get("INKLING_BATCH_MAX_LANES", "64"))
MEM_BUDGET_GIB = float(os.environ.get("INKLING_BATCH_MAX_MEM_GB", "48"))

MAX_PROMPTS = 256
MAX_PROMPT_CHARS = 400_000
MAX_TOTAL_CHARS = 4_000_000
MAX_SYSTEM_CHARS = 20_000
MAX_TOKENS_CAP = 8192
DEFAULT_MAX_TOKENS = 512
MAX_DEADLINE_S = 900.0
DEFAULT_DEADLINE_S = 300.0
CHARS_PER_TOKEN = 3.0          # deliberately pessimistic for code and markdown
SWA_MIB_PER_LANE = 160.0       # sliding-window layers: 1,024 tokens x 160 KiB (erisinky README)
FULL_KIB_PER_TOKEN = 32.0      # global layers (erisinky README)
FILE_MAX_BYTES = 2_000_000
THINKING = ("off", "low", "medium", "high")
# Benchmarks wait on the GPU and show almost no CPU, so they are flagged whenever they exist. Servers sit idle with
# the model loaded, so they are flagged only when they use real CPU.
BENCH_NAMES = re.compile(r"llama-bench|llama-cli|llama-perplexity|bench-batch|inkling bench|inkling-ikv")
SERVER_NAMES = re.compile(r"llama-server|mlx_lm|ollama|/inkling( |$)|Unsloth")
MIN_SHARD_CHARS = 500
ANSWER_LINE = re.compile(r"^[ \t>*_`#-]*ANSWER[ \t*_`]*:[ \t*_`]*(.+?)[ \t*_`]*$", re.I | re.M)

CAVEAT = ("Output is from a small local model and is unverified: check it against real state before relying on it. "
          "Lane output is not bit-identical to a solo run.")


def _log(msg: str) -> None:
    sys.stderr.write(f"[inkling-batch] {msg}\n")
    sys.stderr.flush()


class ToolInputError(Exception):
    pass


class BackendError(Exception):
    pass


# --------------------------------------------------------------------------- configuration

def _workspace() -> str:
    return os.path.realpath(os.environ.get("INKLING_WORKSPACE") or os.getcwd())


def _bin() -> str:
    return os.environ.get("INKLING_BATCH_BIN") or os.path.join(
        _workspace(), "tools", "erisinky-batch", "target", "release", "inkling")


def _model() -> str:
    return os.environ.get("INKLING_BATCH_MODEL") or DEFAULT_MODEL


# --------------------------------------------------------------------------- input validation

def _int(args: dict, name: str, default: int, lo: int, hi: int) -> int:
    v = args.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ToolInputError(f"{name} must be an integer")
    if not lo <= v <= hi:
        raise ToolInputError(f"{name} must be between {lo} and {hi}")
    return v


def _num(args: dict, name: str, default: float, lo: float, hi: float, lo_open: bool = False) -> float:
    v = args.get(name, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ToolInputError(f"{name} must be a number")
    if (v <= lo if lo_open else v < lo) or v > hi:
        raise ToolInputError(f"{name} must be in {'(' if lo_open else '['}{lo}, {hi}]")
    return float(v)


def _str(args: dict, name: str, limit: int, required: bool = False) -> str | None:
    v = args.get(name)
    if v is None:
        if required:
            raise ToolInputError(f"{name} is required")
        return None
    if not isinstance(v, str):
        raise ToolInputError(f"{name} must be a string")
    if required and not v.strip():
        raise ToolInputError(f"{name} must not be empty")
    if len(v) > limit:
        raise ToolInputError(f"{name} is {len(v)} characters; the limit is {limit}")
    return v


def _sampling(args: dict, default_max_tokens: int = DEFAULT_MAX_TOKENS, default_thinking: str = "off") -> dict:
    thinking = args.get("thinking", default_thinking)
    if thinking not in THINKING:
        raise ToolInputError(f"thinking must be one of {', '.join(THINKING)}")
    temp = args.get("temperature", 1.0)
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not 0 < temp <= 2:
        raise ToolInputError("temperature must be in (0, 2]; greedy decoding loops on this model, so 0 is refused")
    seed = args.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ToolInputError("seed must be a non-negative integer")
    return {
        "max_tokens": _int(args, "max_tokens", default_max_tokens, 1, MAX_TOKENS_CAP),
        "thinking": thinking,
        "temperature": float(temp),
        "top_p": _num(args, "top_p", 0.95, 0.0, 1.0, lo_open=True),
        "top_k": _int(args, "top_k", 64, 1, 64),   # the engine's GPU sampler clamps top-k to 64
        "seed": seed,
        "deadline_s": _num(args, "deadline_s", DEFAULT_DEADLINE_S, 1.0, MAX_DEADLINE_S),
        "include_reasoning": bool(args.get("include_reasoning", False)),
        "max_output_chars": _int(args, "max_output_chars", 6000, 200, 100_000),
    }


def _read_workspace_file(rel: str) -> str:
    """Read a text file inside the workspace, refusing escapes, secret-named files, binaries and huge files."""
    ws = _workspace()
    path = os.path.realpath(os.path.join(ws, rel))
    if path != ws and not path.startswith(ws + os.sep):
        raise ToolInputError("file must be inside the workspace")
    relp = os.path.relpath(path, ws)
    if any(_SECRET_NAME.search(part) for part in relp.split(os.sep)):
        raise ToolInputError("file name looks like a secret and is refused")
    if not os.path.isfile(path):
        raise ToolInputError(f"no such file: {rel}")
    if os.path.getsize(path) > FILE_MAX_BYTES:
        raise ToolInputError(f"file is larger than {FILE_MAX_BYTES} bytes")
    with open(path, "rb") as f:
        raw = f.read()
    if b"\0" in raw[:8192]:
        raise ToolInputError("file looks binary")
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- machine state

def _power_state() -> dict:
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {"known": False, "low_power": False}
    low = False
    for line in out.splitlines():
        m = re.match(r"\s*(lowpowermode|powermode)\s+(\d+)", line)
        if m and ((m.group(1) == "lowpowermode" and m.group(2) == "1") or (m.group(1) == "powermode" and m.group(2) == "1")):
            low = True
    return {"known": True, "low_power": low}


def _busy_processes() -> list[str]:
    """GPU-heavy neighbours: other Inkling/llama/mlx processes using real CPU. A hint, not a lock."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,pcpu=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or str(os.getpid()) == parts[0]:
            continue
        try:
            cpu = float(parts[1])
        except ValueError:
            continue
        cmd = parts[2]
        if "inkling_batch_mcp" in cmd or "inkling_fast_mcp" in cmd or "inkling_mcp" in cmd:
            continue
        if BENCH_NAMES.search(cmd) or (cpu >= 5.0 and SERVER_NAMES.search(cmd)):
            found.append(f"pid {parts[0]} {cpu:.0f}% {cmd[:90]}")
    return found[:5]


# --------------------------------------------------------------------------- engine

def _gpu_lease(wait_s: float):
    """Take a shared GPU lease (experiments/_lib/gpulease.py) for one engine call. Returns (info, context).

    Waits up to wait_s while an exclusive job (a benchmark) holds or has requested the GPU; then
    raises BackendError naming it. Without the lease module the call runs unleased."""
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from _lib.gpulease import LeaseTimeout, lease
    except ImportError:
        import contextlib
        return None, contextlib.nullcontext()
    cm = lease("inkling-batch-mcp", exclusive=False, timeout=wait_s)
    try:
        return cm.__enter__(), cm
    except LeaseTimeout:
        from _lib.gpulease import status
        st = status()
        who = [(h["name"], h["mode"]) for h in st["holders"]] + [(d["name"], "waiting") for d in st["drain_requests"]]
        raise BackendError(f"the GPU is reserved by an exclusive job {who}; waited {wait_s:g} s. Retry later.")


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def run_engine(prompts: list[str], sp: dict, *, system: str | None = None, lanes: int | None = None,
               shared_n: int | None = None) -> tuple[list[dict], dict, dict]:
    """Run one batch. Returns (rows sorted by prompt index, engine summary, call meta). Raises BackendError."""
    binary, model = _bin(), _model()
    if not os.path.isfile(binary):
        raise BackendError(f"engine binary not found: {binary}. Build upstream erisinky and apply "
                           f"erisinky-batch-json.patch (see the README), or set INKLING_BATCH_BIN.")
    if not os.path.isfile(model):
        raise BackendError(f"model not found: {model}. Set INKLING_BATCH_MODEL (erisinky needs LM Studio's Q4_K_M file).")
    power = _power_state()
    if power.get("low_power"):
        raise BackendError("macOS Low Power Mode is on, which halves GPU throughput and would make timings meaningless. "
                           "Turn it off and retry.")
    warnings = []
    busy = _busy_processes()
    if busy:
        warnings.append("other GPU-heavy processes are running, so speed will be lower: " + "; ".join(busy))

    n_lanes_req = shared_n if shared_n else (lanes or min(len(prompts), 16))
    sys_tokens = _est_tokens(system or "")
    need = max(_est_tokens(p) for p in prompts) + sys_tokens + 80 + sp["max_tokens"]
    if need > CTX_CAP:
        raise ToolInputError(f"the longest prompt plus max_tokens is about {need} tokens; the cap is {CTX_CAP} "
                             f"(INKLING_BATCH_CTX_CAP). Shorten the input or use batch_split_ask.")
    ctx = min(CTX_CAP, max(1024, math.ceil(need * 1.2 / 64) * 64))
    per_lane_gib = (min(need, 1024) * SWA_MIB_PER_LANE / 1024 + need * FULL_KIB_PER_TOKEN / 1024) / 1024 * 1.25
    fit = max(1, int((MEM_BUDGET_GIB - 1.0) / max(per_lane_gib, 0.01)))
    if shared_n:
        if shared_n > fit:
            raise ToolInputError(f"{shared_n} samples of about {need} tokens need about {shared_n * per_lane_gib:.1f} GiB; "
                                 f"the budget allows {fit} (INKLING_BATCH_MAX_MEM_GB)")
        lanes_used = shared_n
    else:
        lanes_used = max(1, min(n_lanes_req, MAX_LANES, fit))
    mem_gib = min(MEM_BUDGET_GIB, max(2.0, math.ceil(lanes_used * per_lane_gib + 1.0)))

    with tempfile.TemporaryDirectory(prefix="inkling-batch-") as tmp:
        inp, outp = os.path.join(tmp, "in.txt"), os.path.join(tmp, "out.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            f.write("\n".join(_esc(p) for p in prompts) + "\n")
        cmd = [binary, "batch-json", "-m", model, "--prompts-esc", inp, "--out", outp, "-b", str(lanes_used),
               "-n", str(sp["max_tokens"]), "-c", str(ctx), "--max-mem", f"{mem_gib:g}",
               "--temp", f"{sp['temperature']:g}", "--top-p", f"{sp['top_p']:g}", "--top-k", str(sp["top_k"])]
        if sp["seed"] is not None:
            cmd += ["--seed", str(max(1, sp["seed"]))]
        if system:
            cmd += ["--system", system]
        if sp["thinking"] == "off":
            cmd += ["--think-off", "--effort", "none"]
        else:
            cmd += ["--effort", sp["thinking"]]
        if shared_n:
            cmd += ["--shared"]
        held, lease_cm = _gpu_lease(sp["deadline_s"])
        if held and held.get("waited_s"):
            warnings.append(f"waited {held['waited_s']:g} s for the GPU lease")
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                    start_new_session=True)
            try:
                _, stderr = proc.communicate(timeout=sp["deadline_s"])
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.communicate()
                raise BackendError(f"the batch did not finish within {sp['deadline_s']:g} s and was stopped. Use fewer "
                                   f"or shorter prompts, a smaller max_tokens, or a larger deadline_s.")
        finally:
            lease_cm.__exit__(None, None, None)
        wall = time.time() - t0
        rows: list[dict] = []
        summary: dict = {}
        fatal = None
        if os.path.isfile(outp):
            with open(outp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "summary" in obj:
                        summary = obj["summary"]
                    elif "fatal" in obj:
                        fatal = obj["fatal"]
                    else:
                        rows.append(obj)
        if fatal:
            raise BackendError(f"engine refused the batch: {fatal}")
        if proc.returncode != 0 or not summary:
            tail = (stderr or "").strip().splitlines()[-3:]
            raise BackendError(f"engine failed (exit {proc.returncode}): " + " | ".join(tail)[:400])

    rows.sort(key=lambda r: r["i"])
    gen_s = max(summary.get("gen_ms", 0.0) / 1000.0, 1e-9)
    pre_s = max(summary.get("prompt_ms", 0.0) / 1000.0, 1e-9)
    meta = {
        "lanes_used": lanes_used, "ctx": ctx, "wall_s": round(wall, 2),
        "prompt_tokens": summary.get("prompt_tokens"), "gen_tokens": summary.get("gen_tokens"),
        "prefill_tok_s": round(summary.get("prompt_tokens", 0) / pre_s) if summary.get("prompt_tokens") else None,
        "decode_tok_s_aggregate": round(summary.get("gen_tokens", 0) / gen_s) if summary.get("gen_tokens") else None,
        "peak_batch": summary.get("peak_batch"), "kv_mib": summary.get("kv_mib"),
        "sampling": {k: sp[k] for k in ("temperature", "top_p", "top_k", "thinking", "max_tokens")},
    }
    if warnings:
        meta["warnings"] = warnings
    return rows, summary, meta


def _shape_row(r: dict, sp: dict) -> dict:
    if "error" in r:
        return {"i": r["i"], "error": r["error"], "prompt_tokens": r.get("prompt_tokens")}
    content = r.get("content", "")
    lim = sp["max_output_chars"]
    out = {"i": r["i"], "content": content[:lim], "finish": r["finish"],
           "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"]}
    if len(content) > lim:
        out["content_truncated_chars"] = len(content) - lim
    if r["finish"] == "length":
        out["note"] = "stopped at max_tokens; raise max_tokens or lower thinking"
    if sp["include_reasoning"]:
        out["reasoning"] = r.get("reasoning", "")[:lim]
    elif r.get("reasoning_tokens"):
        out["reasoning_tokens"] = r["reasoning_tokens"]
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip(".").lower())


def _extract_answer(text: str, pattern: re.Pattern | None) -> str | None:
    rx = pattern or ANSWER_LINE
    matches = list(rx.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return (m.group(1) if m.groups() else m.group(0)).strip()


def _tally(texts: list[str], pattern: re.Pattern | None) -> dict:
    answers = [_extract_answer(t, pattern) for t in texts]
    keyed: dict[str, list[str]] = {}
    for a in answers:
        if a:
            keyed.setdefault(_norm(a), []).append(a)
    counts = Counter({k: len(v) for k, v in keyed.items()})
    out: dict[str, Any] = {"answers": answers, "unparsed": sum(1 for a in answers if not a)}
    if counts:
        top, n = counts.most_common(1)[0]
        tied = [k for k, c in counts.items() if c == n]
        out.update(majority=keyed[top][0], majority_votes=n, of=len(answers), tied=len(tied) > 1,
                   votes={keyed[k][0]: c for k, c in counts.most_common()})
    else:
        out.update(majority=None, majority_votes=0, of=len(answers), tied=False, votes={})
    return out


# --------------------------------------------------------------------------- tools

def batch_map(args: dict) -> tuple[Any, dict]:
    prompts = args.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ToolInputError("prompts must be a non-empty list of strings")
    if len(prompts) > MAX_PROMPTS:
        raise ToolInputError(f"at most {MAX_PROMPTS} prompts per call")
    if not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ToolInputError("every prompt must be a non-empty string")
    if any(len(p) > MAX_PROMPT_CHARS for p in prompts) or sum(len(p) for p in prompts) > MAX_TOTAL_CHARS:
        raise ToolInputError(f"a prompt is over {MAX_PROMPT_CHARS} characters or the total is over {MAX_TOTAL_CHARS}")
    system = _str(args, "system", MAX_SYSTEM_CHARS)
    sp = _sampling(args)
    lanes = _int(args, "lanes", min(len(prompts), 16), 1, MAX_LANES)
    rows, _summary, meta = run_engine(prompts, sp, system=system, lanes=lanes)
    return [_shape_row(r, sp) for r in rows], meta


def batch_samples(args: dict) -> tuple[Any, dict]:
    prompt = _str(args, "prompt", MAX_PROMPT_CHARS, required=True)
    system = _str(args, "system", MAX_SYSTEM_CHARS)
    n = _int(args, "n", 8, 2, MAX_LANES)
    sp = _sampling(args)
    pattern = None
    rx = _str(args, "answer_regex", 200)
    if rx:
        try:
            pattern = re.compile(rx, re.I | re.M)
        except re.error as e:
            raise ToolInputError(f"answer_regex does not compile: {e}")
    rows, _summary, meta = run_engine([prompt], sp, system=system, shared_n=n)
    shaped = [_shape_row(r, sp) for r in rows]
    result: dict[str, Any] = {"samples": shaped}
    texts = [r.get("content", "") for r in rows]
    if pattern is not None or args.get("tally", False):
        result["tally"] = _tally(texts, pattern)
    meta["note"] = "one shared prefill; lanes differ only by sampling"
    return result, meta


def _split_shards(text: str, k: int) -> list[str]:
    target = max(1, math.ceil(len(text) / k))
    pieces: list[str] = []
    for p in re.split(r"\n[ \t]*\n", text):
        p = p.strip("\n")
        if not p.strip():
            continue
        while len(p) > target * 1.5:
            cut = p.rfind("\n", 0, target)
            if cut < target * 0.5:
                cut = p.rfind(" ", 0, target)
            if cut < target * 0.5:
                cut = target
            pieces.append(p[:cut])
            p = p[cut:].lstrip("\n ")
        pieces.append(p)
    shards: list[str] = []
    cur: list[str] = []
    size = 0
    for piece in pieces:
        cur.append(piece)
        size += len(piece) + 2
        if size >= target and len(shards) < k - 1:
            shards.append("\n\n".join(cur))
            cur, size = [], 0
    if cur:
        shards.append("\n\n".join(cur))
    return shards


R1_TEMPLATE = (
    "You are reading part {i} of {k} of a longer document. Your only job is to extract facts.\n"
    "A question will be answered later by someone who sees only the facts you report. Do NOT answer it.\n"
    "Question: {question}\n\n"
    "List, one per line, every statement in this part that could bear on the question, as a short verbatim quote or a "
    "close paraphrase. Include facts that look irrelevant but might matter, and keep any figures and names exactly. "
    "Do not draw conclusions and do not repeat what other parts might say. If nothing in this part is relevant, "
    "write NONE.\n\n--- PART {i} OF {k} ---\n{shard}\n--- END OF PART {i} ---")

R2_TEMPLATE = (
    "Below are facts extracted from {k} parts of a document. Each part was read separately, so no single reader saw "
    "all of it. Treat remarks about what the answer 'probably' is as opinion; rely only on the stated facts.\n\n"
    "{facts}\n\nQuestion: {question}\n\n"
    "Reason from the facts above only. If they do not determine an answer, say what is missing. End with one final "
    "line of the form `ANSWER: <short answer>`.")


def batch_split_ask(args: dict) -> tuple[Any, dict]:
    question = _str(args, "question", 5000, required=True)
    text, rel = _str(args, "text", MAX_TOTAL_CHARS), _str(args, "file", 1000)
    if bool(text) == bool(rel):
        raise ToolInputError("give exactly one of text or file")
    if rel:
        text = _read_workspace_file(rel)
    assert text is not None
    shards_n = _int(args, "shards", 4, 2, 16)
    answers_n = _int(args, "answers", 4, 1, MAX_LANES)
    sp = _sampling(args, default_max_tokens=900, default_thinking="low")
    sp_r1 = dict(sp, max_tokens=_int(args, "fact_tokens", 400, 50, 2000), thinking="off", include_reasoning=False)
    shards_n = min(shards_n, len(text.strip()) // MIN_SHARD_CHARS)
    if shards_n < 2:
        raise ToolInputError(f"the text is too short to split (under {2 * MIN_SHARD_CHARS} characters); "
                             f"ask it directly with batch_samples or batch_map")
    shards = _split_shards(text, shards_n)
    if len(shards) < 2:
        raise ToolInputError("the text is too short to split; ask it directly with batch_samples or batch_map")
    k = len(shards)
    r1_prompts = [R1_TEMPLATE.format(i=i + 1, k=k, question=question, shard=s) for i, s in enumerate(shards)]
    rows1, _s1, meta1 = run_engine(r1_prompts, sp_r1, lanes=k)
    facts = []
    for i, r in enumerate(rows1):
        if "error" in r:
            raise BackendError(f"shard {i + 1} was refused by the engine ({r['error']}); use more shards")
        facts.append(r.get("content", "").strip() or "NONE")
    pooled = "\n\n".join(f"[Part {i + 1} of {k}]\n{f}" for i, f in enumerate(facts))
    r2_prompt = R2_TEMPLATE.format(k=k, facts=pooled, question=question)
    if answers_n == 1:
        rows2, _s2, meta2 = run_engine([r2_prompt], sp, lanes=1)
    else:
        rows2, _s2, meta2 = run_engine([r2_prompt], sp, shared_n=answers_n)
    texts = [r.get("content", "") for r in rows2]
    tally = _tally(texts, None)
    lim = sp["max_output_chars"]
    result = {
        "answer": tally["majority"],
        "votes": tally["votes"], "majority_votes": tally["majority_votes"], "of": tally["of"],
        "tied": tally["tied"], "unparsed": tally["unparsed"],
        "shards": [{"part": i + 1, "chars": len(s), "facts": f[:lim], "finish": rows1[i]["finish"]}
                   for i, (s, f) in enumerate(zip(shards, facts))],
        "first_full_answer": texts[0][:lim] if texts else "",
        "experimental": ("Fact forwarding across lanes. Whether this beats reading the whole text in one lane is being "
                         "tested in experiments/inkling-fact-forwarding and is NOT established. On a text that fits "
                         "one lane, splitting did not shorten prefill; compare against batch_samples on the full text."),
    }
    meta = {"round1_extract": meta1, "round2_answer": meta2,
            "wall_s": round(meta1["wall_s"] + meta2["wall_s"], 2),
            "shard_count": k, "answers": answers_n}
    return result, meta


def batch_status(_args: dict) -> tuple[Any, dict]:
    binary, model = _bin(), _model()
    info: dict[str, Any] = {
        "engine": binary, "engine_exists": os.path.isfile(binary),
        "model": model, "model_exists": os.path.isfile(model),
        "model_gib": round(os.path.getsize(model) / 2**30, 1) if os.path.isfile(model) else None,
        "ctx_cap": CTX_CAP, "max_lanes": MAX_LANES, "mem_budget_gib": MEM_BUDGET_GIB,
        "power": _power_state(), "busy_processes": _busy_processes(),
    }
    if info["engine_exists"]:
        try:
            helptext = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=20)
            info["supports_batch_json"] = "batch-json" in (helptext.stdout + helptext.stderr)
        except (OSError, subprocess.SubprocessError):
            info["supports_batch_json"] = None
    info["note"] = ("Aggregate decode grows with lanes (about 120 tok/s at 4, 200 at 8, 340 at 16, 830-910 at 64 on "
                    "an M5 Max) but each lane slows down; prefill does not parallelise.")
    return info, {}


TOOLS = [
    {
        "name": "batch_map",
        "description": (
            "Run N independent prompts through the local Inkling 2 Mini, one lane each, and return one result per prompt "
            "in order. Raises total throughput (hundreds of tok/s across lanes) for classify/extract/summarise-style "
            "work over many items; each answer is no faster than a single call, and long prompts are prefill-bound "
            "(about 3.5-4k tok/s in total). Thinking is off by default. Output is unverified."),
        "inputSchema": {"type": "object", "additionalProperties": False, "required": ["prompts"], "properties": {
            "prompts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_PROMPTS,
                        "description": "Independent user prompts."},
            "system": {"type": "string", "description": "Optional system prompt shared by every prompt."},
            "max_tokens": {"type": "integer", "default": DEFAULT_MAX_TOKENS, "description": f"Per-prompt cap, 1..{MAX_TOKENS_CAP}."},
            "thinking": {"type": "string", "enum": list(THINKING), "default": "off"},
            "temperature": {"type": "number", "default": 1.0, "description": "In (0, 2]; greedy is refused (it loops)."},
            "top_p": {"type": "number", "default": 0.95}, "top_k": {"type": "integer", "default": 64, "description": "1..64."},
            "seed": {"type": "integer", "description": "Optional. Results also depend on lane layout."},
            "lanes": {"type": "integer", "description": f"Concurrent lanes, 1..{MAX_LANES}. Default min(prompts, 16); lowered automatically to fit GPU memory."},
            "include_reasoning": {"type": "boolean", "default": False},
            "max_output_chars": {"type": "integer", "default": 6000},
            "deadline_s": {"type": "number", "default": DEFAULT_DEADLINE_S, "description": f"Wall-clock limit, up to {MAX_DEADLINE_S:g}."}}},
    },
    {
        "name": "batch_samples",
        "description": (
            "Draw N samples of ONE prompt in lanes from a single shared prefill (self-consistency, majority vote, "
            "pass@n). With answer_regex (one capture group) or tally=true (looks for a final `ANSWER: x` line) it also "
            "returns the vote counts and the majority answer. Cheaper than N separate calls because the prompt is "
            "prefilled once. Output is unverified; agreement between samples is not correctness."),
        "inputSchema": {"type": "object", "additionalProperties": False, "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}, "system": {"type": "string"},
            "n": {"type": "integer", "default": 8, "description": f"Samples, 2..{MAX_LANES}."},
            "answer_regex": {"type": "string", "description": "Python regex; group 1 (or the whole match) is the answer; the last match wins."},
            "tally": {"type": "boolean", "default": False, "description": "Tally the final `ANSWER:` lines."},
            "max_tokens": {"type": "integer", "default": DEFAULT_MAX_TOKENS},
            "thinking": {"type": "string", "enum": list(THINKING), "default": "off"},
            "temperature": {"type": "number", "default": 1.0}, "top_p": {"type": "number", "default": 0.95},
            "top_k": {"type": "integer", "default": 64}, "seed": {"type": "integer"},
            "include_reasoning": {"type": "boolean", "default": False},
            "max_output_chars": {"type": "integer", "default": 6000}, "deadline_s": {"type": "number", "default": DEFAULT_DEADLINE_S}}},
    },
    {
        "name": "batch_split_ask",
        "description": (
            "EXPERIMENTAL. Answer a question about a long text by splitting it into shards, extracting the relevant facts "
            "from each shard in parallel lanes (round 1, no answering), then answering from the pooled facts with a "
            "vote across lanes (round 2). Based on the 'forward the facts, not the conclusions' result from the "
            "Recursive Collective Intelligence paper. NOT shown to beat reading the whole text in one lane, and it does "
            "not shorten prefill; use it to test, not as a default. Give `text` or a workspace `file`."),
        "inputSchema": {"type": "object", "additionalProperties": False, "required": ["question"], "properties": {
            "question": {"type": "string"}, "text": {"type": "string"},
            "file": {"type": "string", "description": "Path inside the workspace (secret-named and binary files are refused)."},
            "shards": {"type": "integer", "default": 4, "description": "2..16 paragraph-aligned shards."},
            "answers": {"type": "integer", "default": 4, "description": f"Round-2 lanes that vote, 1..{MAX_LANES}."},
            "fact_tokens": {"type": "integer", "default": 400, "description": "Round-1 cap per shard."},
            "max_tokens": {"type": "integer", "default": 900, "description": "Round-2 cap per answer."},
            "thinking": {"type": "string", "enum": list(THINKING), "default": "low", "description": "Round 2 only; round 1 is always off."},
            "temperature": {"type": "number", "default": 1.0}, "top_p": {"type": "number", "default": 0.95},
            "top_k": {"type": "integer", "default": 64}, "seed": {"type": "integer"},
            "max_output_chars": {"type": "integer", "default": 6000}, "deadline_s": {"type": "number", "default": DEFAULT_DEADLINE_S, "description": "Per round."}}},
    },
    {
        "name": "batch_status",
        "description": "Engine binary, model, patch support, power state (Low Power Mode refuses runs) and other GPU-heavy processes. Read-only.",
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
]

HANDLERS = {"batch_map": batch_map, "batch_samples": batch_samples, "batch_split_ask": batch_split_ask, "batch_status": batch_status}


def _text_result(blocks: list[str], is_error: bool = False) -> dict:
    out: dict[str, Any] = {"content": [{"type": "text", "text": b} for b in blocks]}
    if is_error:
        out["isError"] = True
    return out


def call_tool(name: str, args: dict) -> dict:
    handler = HANDLERS.get(name)
    if handler is None:
        return _text_result([f"unknown tool: {name}"], True)
    try:
        result, meta = handler(args)
        blocks = [json.dumps(result, ensure_ascii=False, indent=1 if name == "batch_status" else None,
                             separators=None if name == "batch_status" else (",", ":"))]
        if meta:
            meta = dict(meta, caveat=CAVEAT)
            blocks.append(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
        return _text_result(blocks)
    except ToolInputError as e:
        return _text_result([f"invalid input: {e}"], True)
    except BackendError as e:
        return _text_result([str(e)], True)


def _dispatch(method: str, params: dict) -> dict:
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[-1]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Local lane-batched Inkling 2 Mini (erisinky on this Mac). Use it for many independent items or many "
                "samples of one prompt: total throughput rises with lanes, but each answer is not faster and prefill is "
                "serial. Output is unverified. It shares the GPU with other local model processes."),
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return _text_result(["arguments must be an object"], True)
        return call_tool(str(params.get("name")), args)
    raise LookupError(method)


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    _log(f"starting; engine {_bin()}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if not isinstance(msg, dict) or "method" not in msg or msg.get("id") is None:
            continue
        try:
            _write({"jsonrpc": "2.0", "id": msg["id"], "result": _dispatch(msg["method"], msg.get("params") or {})})
        except LookupError:
            _write({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"method not found: {msg['method']}"}})
        except Exception as e:  # keep the server alive on unexpected errors
            _log(f"internal error: {e!r}")
            _write({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message": f"internal error: {e}"}})


if __name__ == "__main__":
    main()
