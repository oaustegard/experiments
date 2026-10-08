#!/usr/bin/env python3
"""Assess Claude transcripts for compliance, task type, friction and success.

Pipeline: source transcript -> normalized events -> deterministic signals +
compact digest -> one Claude Haiku 5.5 call per transcript (structured output)
-> merged record (model judgment + signals + arithmetic done in code).

Sources understood by `load_events`:
  * Claude Code session JSONL (~/.claude/projects/*/*.jsonl)
  * Compliance API session transcript  (/v1/compliance/apps/sessions/{local,remote}/{id}/messages)
  * Compliance API chat                 (/v1/compliance/apps/chats/{id}/messages)

Subcommands:
  digest        SRC... --out DIR      write <id>.digest.md + <id>.signals.json
  emit-prompts  DIGEST_DIR --out DIR  one self-contained prompt file per transcript
                                      (for dispatch through a harness with no API key)
  batch-submit  DIGEST_DIR            submit a Message Batch (needs ANTHROPIC_API_KEY)
  batch-collect BATCH_ID --digests DIR --out results.jsonl
  merge         RAW_DIR --digests DIR --out results.jsonl   (raw model JSON files -> records)
  report        results.jsonl         aggregate tables
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "claude-haiku-5-5"
ESCALATION_MODEL = "claude-sonnet-5-5"   # rerun target for Haiku refusals
DIGEST_CHAR_BUDGET = 240_000             # ~60-80K tokens: stays under Haiku's 100K price break

# ---------------------------------------------------------------- events


@dataclass
class Event:
    role: str            # user | assistant
    kind: str            # text | tool_use | tool_result | marker
    text: str = ""
    tool: str = ""
    is_error: bool = False
    ts: str | None = None
    human: bool = False  # a user text the person typed (not harness / tool output)


@dataclass
class Transcript:
    id: str
    source: str
    title: str = ""
    events: list[Event] = field(default_factory=list)


HARNESS_PREFIXES = (
    "<task-notification>", "<local-command", "<command-name>", "<command-message>",
    "Stop hook feedback", "[Image:", "<system-reminder>", "Caveat:",
)


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
        elif isinstance(b, str):
            out.append(b)
    return "\n".join(out)


def _tool_input_str(inp) -> str:
    if isinstance(inp, str):          # compliance API: JSON-encoded string, maybe truncated
        try:
            inp = json.loads(inp)
        except ValueError:
            return inp
    if not isinstance(inp, dict):
        return str(inp)
    for k in ("command", "file_path", "path", "pattern", "url", "query", "prompt", "description"):
        if k in inp:
            return f"{k}={inp[k]}"
    return json.dumps(inp, ensure_ascii=False)


def _blocks_to_events(role, content, ts, human_ok: bool) -> list[Event]:
    evs = []
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for b in content or []:
        t = b.get("type")
        if t == "text":
            txt = b.get("text", "")
            human = role == "user" and human_ok and not txt.lstrip().startswith(HARNESS_PREFIXES)
            evs.append(Event(role, "text", txt, ts=ts, human=human))
        elif t == "tool_use":
            evs.append(Event(role, "tool_use", _tool_input_str(b.get("input")), tool=b.get("name", ""), ts=ts))
        elif t == "tool_result":
            evs.append(Event(role, "tool_result", _block_text(b.get("content")), tool=b.get("name", ""),
                             is_error=bool(b.get("is_error")), ts=ts))
    return evs


def load_events(path: str) -> Transcript:
    p = Path(path)
    raw = p.read_text(errors="replace")
    stripped = raw.lstrip()
    # Compliance API responses are single JSON documents.
    if stripped.startswith("{") and "\n{" not in stripped[:2000].strip()[1:]:
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        if isinstance(doc, dict) and "chat_messages" in doc:
            tr = Transcript(doc.get("id", p.stem), "compliance_chat", doc.get("name") or "")
            for m in doc["chat_messages"]:
                tr.events += _blocks_to_events(m.get("role"), m.get("content"), m.get("created_at"), True)
                for f in (m.get("files") or []):
                    tr.events.append(Event(m.get("role"), "marker", f"[attached file: {f.get('filename')}]"))
            return tr
        if isinstance(doc, dict) and "data" in doc and "session" in doc:
            s = doc["session"]
            tr = Transcript(s.get("id", p.stem), "compliance_session:" + str(s.get("product_surface")))
            for m in doc["data"]:
                prov = m.get("provenance") or {}
                if prov.get("type") == "synthetic_marker" or m.get("content_unavailable"):
                    tr.events.append(Event(m.get("role"), "marker", _block_text(m.get("content")) or "[content unavailable]"))
                    continue
                tr.events += _blocks_to_events(m.get("role"), m.get("content"), m.get("created_at"), True)
            return tr
    # Claude Code JSONL
    tr = Transcript(p.stem, "claude_code_jsonl")
    for line in raw.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        typ = d.get("type")
        if typ in ("custom-title", "ai-title") and not tr.title:
            tr.title = d.get("customTitle") or d.get("aiTitle") or ""
        if typ == "system" and d.get("subtype") == "compact_boundary":
            tr.events.append(Event("user", "marker", "[context compacted]", ts=d.get("timestamp")))
        if typ not in ("user", "assistant") or d.get("isSidechain"):
            continue
        if d.get("isCompactSummary"):
            tr.events.append(Event("user", "marker", "[compaction summary omitted]", ts=d.get("timestamp")))
            continue
        msg = d.get("message") or {}
        human_ok = typ == "user" and not d.get("isMeta")
        evs = _blocks_to_events(msg.get("role", typ), msg.get("content"), d.get("timestamp"), human_ok)
        if d.get("isMeta"):
            evs = [e for e in evs if e.kind != "text"]
        tr.events += evs
    return tr

# ---------------------------------------------------------------- signals

SECRET_PATTERNS = {
    "anthropic_key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "openai_key": r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}",
    "github_token": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}",
    "aws_access_key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "jwt": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "bearer_literal": r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._~+/-]{24,}",
    "password_assignment": r"(?i)\b(?:password|passwd|pwd)\s*[=:]\s*['\"][^'\"\s]{6,}['\"]",
    # ENV_STYLE_TOKEN=<value with a digit and a lowercase letter>: skips identifiers and $(...) lookups
    "secret_assignment": r"\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|API_?KEY|PASSWORD|PASSWD)\b\s*[=:]\s*['\"]?(?=[A-Za-z0-9_\-./+]*\d)(?=[A-Za-z0-9_\-./+]*[a-z])[A-Za-z0-9_\-./+]{24,}",
}
PII_PATTERNS = {
    "us_ssn": r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    # grouped 4-4-4-4 / 4-6-5, or a bare 15-16 digit run with a card-network prefix
    "credit_card": r"(?<![\d.-])(?:[3-6]\d{3}[ -]\d{4}[ -]\d{4}[ -]\d{4}|3[47]\d{2}[ -]\d{6}[ -]\d{5}|(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13}|6011\d{12}))(?![\d.-])",
}
RISKY_COMMANDS = {
    "rm_rf": r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r",
    "force_push": r"git\s+push\b[^\n]*(?:--force\b|\s-f\b)",
    "history_rewrite": r"git\s+(?:reset\s+--hard|filter-branch|rebase\s+-i)",
    "skip_verification": r"--no-verify\b|dangerouslyDisableSandbox\"?\s*[:=]\s*true",
    "sql_drop": r"(?i)\b(?:drop\s+(?:table|database)|truncate\s+table)\b",
    "pipe_to_shell": r"curl[^\n|]*\|\s*(?:sudo\s+)?(?:ba)?sh\b",
    "chmod_world": r"chmod\s+(?:-R\s+)?777",
}
DENIAL_MARKERS = (
    "doesn't want to proceed", "user rejected", "permission to use", "was blocked",
    "denied by", "auto mode denied", "permission denied",
)
INTERRUPT_MARKER = "[Request interrupted by user"


def _luhn(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 16:
        return False
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x = x * 2 - 9 if x * 2 > 9 else x * 2
        tot += x
    return tot % 10 == 0


def redact(text: str) -> str:
    for name, pat in SECRET_PATTERNS.items():
        text = re.sub(pat, f"[REDACTED:{name}]", text)
    return text


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_signals(tr: Transcript) -> dict:
    sig = {"secret_hits": {}, "pii_hits": {}, "risky_commands": {}}
    alltext = []
    for e in tr.events:
        alltext.append(e.text)
        if e.kind == "tool_use":
            for name, pat in RISKY_COMMANDS.items():
                if re.search(pat, e.text):
                    sig["risky_commands"][name] = sig["risky_commands"].get(name, 0) + 1
    blob = "\n".join(alltext)
    for name, pat in SECRET_PATTERNS.items():
        n = len(re.findall(pat, blob))
        if n:
            sig["secret_hits"][name] = n
    n_ssn = len(re.findall(PII_PATTERNS["us_ssn"], blob))
    if n_ssn:
        sig["pii_hits"]["us_ssn"] = n_ssn
    n_cc = sum(1 for m in re.findall(PII_PATTERNS["credit_card"], blob) if _luhn(m))
    if n_cc:
        sig["pii_hits"]["credit_card"] = n_cc

    human = [e for e in tr.events if e.human]
    tool_uses = [e for e in tr.events if e.kind == "tool_use"]
    results = [e for e in tr.events if e.kind == "tool_result"]
    errors = [e for e in results if e.is_error]
    sig.update(
        human_turns=len(human),
        tool_calls=len(tool_uses),
        tool_errors=len(errors),
        tool_error_rate=round(len(errors) / len(results), 3) if results else None,
        interrupts=sum(INTERRUPT_MARKER in e.text for e in tr.events),
        permission_denials=sum(any(m in e.text.lower() for m in DENIAL_MARKERS) for e in errors),
        compactions=sum(e.kind == "marker" and "compact" in e.text for e in tr.events),
        tools_used=sorted({e.tool for e in tool_uses}),
    )
    # Time: wall clock, and a crude human-attention proxy = for each typed user
    # turn, the gap since the previous event, capped at 10 min (reading the last
    # output + writing the reply). Unknown timestamps -> None, never 0.
    times = [(_parse_ts(e.ts), e) for e in tr.events]
    stamped = [(t, e) for t, e in times if t]
    if len(stamped) >= 2:
        sig["wall_clock_min"] = round((stamped[-1][0] - stamped[0][0]).total_seconds() / 60, 1)
        att, prev = 0.0, None
        for t, e in stamped:
            if e.human and prev is not None:
                att += min(max((t - prev).total_seconds() / 60, 0.5), 10)
            elif e.human:
                att += 2.0
            prev = t
        sig["human_attention_min"] = round(att, 1)
    else:
        sig["wall_clock_min"] = None
        sig["human_attention_min"] = None
    sig["deterministic_flag"] = bool(sig["secret_hits"] or sig["pii_hits"])
    return sig

# ---------------------------------------------------------------- digest


def _clip(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    h = n * 2 // 3
    return s[:h] + f" …[{len(s) - n} chars cut]… " + s[-(n - h):]


def _secret_hits(text: str) -> list[tuple[str, int]]:
    out = []
    for k, pat in SECRET_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out.append((k, m.start()))
    return out


def render_digest(tr: Transcript, sig: dict) -> str:
    lines, turn = [], 0
    for e0 in tr.events:
        e = Event(e0.role, e0.kind, redact(e0.text), e0.tool, e0.is_error, e0.ts, e0.human)
        if e.human:
            turn += 1
            lines.append((3, f"\n### T{turn} USER ({e.ts or '?'})\n{_clip(e.text, 4000)}"))
        elif e.kind == "marker":
            lines.append((3, f"  {e.text}"))
        elif e.role == "user" and e.kind == "text":
            if INTERRUPT_MARKER in e.text:
                lines.append((3, "  [USER INTERRUPTED]"))
            elif e.text.strip():
                lines.append((1, f"  [harness] {_clip(e.text, 160)}"))
        elif e.kind == "text" and e.text.strip():
            lines.append((2, f"  CLAUDE: {_clip(e.text, 900)}"))
        elif e.kind == "tool_use":
            risky = [k for k, pat in RISKY_COMMANDS.items() if re.search(pat, e0.text)]
            if risky:
                lines.append((3, f"  → {e.tool} [risky-cmd: {', '.join(risky)}]: {_clip(e.text, 400)}"))
            else:
                lines.append((1, f"  → {e.tool}: {_clip(e.text, 220)}"))
        elif e.kind == "tool_result" and e.is_error:
            lines.append((2, f"  ✗ {e.tool or 'tool'} ERROR: {_clip(e.text, 350)}"))
        # after the event line, so a typed message's hit carries its own turn number
        hits = _secret_hits(e0.text)
        if hits:   # make the location visible even when the event itself is not rendered
            where = f"{e.kind} of {e.tool}" if e.tool else ("user message" if e.human else e.kind)
            red = e.text                      # already fully redacted: slice it, never the raw text
            at = max(red.find("[REDACTED:"), 0)
            ctx = red[max(0, at - 160): at + 200].replace("\n", " ")
            lines.append((3, f"  [secret-scan] {', '.join(k for k, _ in hits)} in {where} "
                             f"(turn T{turn}): …{ctx}…"))
    body = "\n".join(t for _, t in lines)
    if len(body) > DIGEST_CHAR_BUDGET:
        # Drop priority-1 lines (tool calls, harness) from the middle out.
        keep = [x for x in lines]
        p1 = [i for i, (p, _) in enumerate(keep) if p == 1]
        mid = len(p1) // 2
        order = sorted(p1, key=lambda i: abs(p1.index(i) - mid))
        dropped = set()
        size = len(body)
        for i in order:
            if size <= DIGEST_CHAR_BUDGET:
                break
            dropped.add(i)
            size -= len(keep[i][1]) + 1
        body = "\n".join(t if i not in dropped else "  …" for i, (_, t) in enumerate(keep))
        body = re.sub(r"(?:\n  …)+", lambda m: f"\n  […{m.group(0).count('…')} tool calls elided…]", body)
        if len(body) > DIGEST_CHAR_BUDGET:
            body = _clip(body, DIGEST_CHAR_BUDGET)
    s = {k: v for k, v in sig.items() if k != "tools_used"}
    head = (
        f"# Transcript {tr.id}\nsource: {tr.source}\n"
        f"signals: {json.dumps(s, sort_keys=True)}\n"
        f"tools_used: {', '.join(sig['tools_used'][:40])}\n"
    )
    return redact(head + body)

# ---------------------------------------------------------------- schema

TASK_CATEGORIES = [
    "software_feature", "bug_fix_debugging", "refactor_migration", "code_review",
    "testing_qa", "devops_infra", "data_analysis", "ml_ai_engineering",
    "agent_prompt_tooling", "automation_scripting", "research_investigation",
    "writing_documentation", "communication_drafting", "creative_content",
    "planning_strategy", "learning_explanation", "admin_configuration", "other",
]
COMPLIANCE_CATEGORIES = [
    "credential_exposure", "sensitive_personal_data", "confidential_data_egress",
    "security_control_bypass", "destructive_or_unreviewed_change", "offensive_security",
    "prompt_injection", "legal_regulatory_hr", "unapproved_service_or_account", "other",
]
FRICTION_TYPES = [
    "misunderstood_request", "wrong_approach", "buggy_output", "tool_or_env_failure",
    "permission_or_access_block", "repeated_correction", "overreach_or_scope_creep",
    "unverified_or_false_claim", "slow_or_stalled", "context_loss", "format_or_verbosity",
    "user_interrupted", "other",
]


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


S, N, B = {"type": "string"}, {"type": "number"}, {"type": "boolean"}


def enum(vals):
    return {"type": "string", "enum": vals}


SCHEMA = _obj({
    "summary": S,
    "task": _obj({
        "categories": {"type": "array", "items": enum(TASK_CATEGORIES)},
        "primary": enum(TASK_CATEGORIES),
        "domain": S,
        "autonomy": enum(["interactive", "delegated", "unattended"]),
    }),
    "compliance": _obj({
        "max_severity": enum(["none", "low", "medium", "high"]),
        "issues": {"type": "array", "items": _obj({
            "category": enum(COMPLIANCE_CATEGORIES),
            "severity": enum(["low", "medium", "high"]),
            "actor": enum(["user_request", "assistant_action", "third_party_content"]),
            "turn": S,
            "evidence": S,
            "handled": enum(["mitigated", "not_mitigated", "unclear"]),
        })},
    }),
    "friction": _obj({
        "level": enum(["none", "low", "moderate", "high"]),
        "events": {"type": "array", "items": _obj({
            "type": enum(FRICTION_TYPES),
            "turn": S,
            "description": S,
            "cost": enum(["minor", "moderate", "major"]),
        })},
        "end_sentiment": enum(["satisfied", "neutral", "frustrated", "unknown"]),
    }),
    "success": _obj({
        "outcome": enum(["achieved", "mostly_achieved", "partial", "not_achieved", "unclear"]),
        "deliverables": {"type": "array", "items": _obj({
            "what": S,
            "evidence_turn": S,
            "role": S,
            "manual_hours_low": N,
            "manual_hours_high": N,
        })},
        "estimate_confidence": enum(["low", "medium", "high"]),
        "estimate_rationale": S,
    }),
})

# ---------------------------------------------------------------- prompting


def system_prompt() -> str:
    return (HERE / "prompt.md").read_text()


def user_message(digest: str) -> str:
    return f"<transcript_digest>\n{digest}\n</transcript_digest>\n\nAssess this transcript."


def finalize(model_json: dict, sig: dict, tid: str, model: str) -> dict:
    """Arithmetic and policy live here, not in the model."""
    d = model_json.get("success", {}).get("deliverables", [])
    lo = sum(float(x.get("manual_hours_low") or 0) for x in d)
    hi = sum(float(x.get("manual_hours_high") or 0) for x in d)
    hi = max(hi, lo)
    att = sig.get("human_attention_min")
    att_h = att / 60 if att is not None else None
    sev = model_json.get("compliance", {}).get("max_severity", "none")
    return {
        "id": tid,
        "model": model,
        "assessment": model_json,
        "signals": sig,
        "derived": {
            "manual_hours_low": round(lo, 2),
            "manual_hours_high": round(hi, 2),
            "human_hours_in_session": round(att_h, 2) if att_h is not None else None,
            # Saved = counterfactual effort minus the person's own time in the session.
            "hours_saved_low": round(max(lo - att_h, 0), 2) if att_h is not None else None,
            "hours_saved_high": round(max(hi - att_h, 0), 2) if att_h is not None else None,
            "flag_for_review": sev in ("medium", "high") or sig.get("deterministic_flag", False),
        },
    }

# ---------------------------------------------------------------- commands


def cmd_digest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for src in a.src:
        for path in sorted(glob.glob(src)) or [src]:
            tr = load_events(path)
            sig = compute_signals(tr)
            if sig["human_turns"] == 0 and not a.keep_empty:
                continue
            tid = tr.id if tr.source != "claude_code_jsonl" else Path(path).stem
            (out / f"{tid}.digest.md").write_text(render_digest(tr, sig))
            (out / f"{tid}.signals.json").write_text(json.dumps(sig, indent=1))
    print(f"{len(list(out.glob('*.digest.md')))} digests in {out}")


def _digests(d):
    return sorted(Path(d).glob("*.digest.md"))


def cmd_emit(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sysp = system_prompt()
    for p in _digests(a.digests):
        tid = p.name[: -len(".digest.md")]
        (out / f"{tid}.prompt.md").write_text(
            f"{sysp}\n\n## Output JSON schema\n```json\n{json.dumps(SCHEMA)}\n```\n\n"
            f"{user_message(p.read_text())}\n\nReply with the JSON object only.")
    print(f"prompts in {out}")


def _request_params(digest: str, model: str = MODEL) -> dict:
    return {
        "model": model,
        "max_tokens": 8000,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(digest)}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def _cid(tid: str) -> str:
    # custom_id max 64 chars, [a-zA-Z0-9_-]
    return re.sub(r"[^A-Za-z0-9_-]", "_", tid)[:48] + "_" + hashlib.sha1(tid.encode()).hexdigest()[:8]


def cmd_batch_submit(a):
    import anthropic
    client = anthropic.Anthropic()
    reqs = []
    for p in _digests(a.digests):
        tid = p.name[: -len(".digest.md")]
        reqs.append({"custom_id": _cid(tid), "params": _request_params(p.read_text(), a.model)})
    b = client.messages.batches.create(requests=reqs)
    print(b.id)


def cmd_batch_collect(a):
    import anthropic
    client = anthropic.Anthropic()
    ids = {_cid(p.name[: -len('.digest.md')]): p for p in _digests(a.digests)}
    escalate = []
    with open(a.out, "a") as fh:
        for r in client.messages.batches.results(a.batch_id):
            p = ids.get(r.custom_id)
            tid = p.name[: -len(".digest.md")] if p else r.custom_id
            sig = json.loads(p.with_suffix("").with_suffix(".signals.json").read_text()) if p else {}
            if r.result.type != "succeeded":
                escalate.append((tid, r.result.type))
                continue
            msg = r.result.message
            if msg.stop_reason == "refusal":       # Haiku 5.5 has no server-side fallback
                escalate.append((tid, f"refusal:{getattr(msg.stop_details, 'category', None)}"))
                continue
            if msg.stop_reason == "max_tokens":
                escalate.append((tid, "max_tokens"))
                continue
            text = next(b.text for b in msg.content if b.type == "text")
            fh.write(json.dumps(finalize(json.loads(text), sig, tid, msg.model)) + "\n")
    for tid, why in escalate:
        print(f"ESCALATE {tid} {why}", file=sys.stderr)
    print(f"collected; {len(escalate)} need a rerun on {ESCALATION_MODEL}")


def _extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    return json.JSONDecoder().raw_decode(text[text.find("{"):])[0]


def cmd_merge(a):
    import jsonschema
    n_ok = n_bad = 0
    with open(a.out, "w") as fh:
        for raw in sorted(Path(a.raw).glob("*.json")):
            tid = raw.stem
            sig = json.loads((Path(a.digests) / f"{tid}.signals.json").read_text())
            try:
                obj = _extract_json(raw.read_text())
                jsonschema.validate(obj, SCHEMA)
            except Exception as e:  # report and skip; never write a half record
                print(f"INVALID {tid}: {str(e)[:200]}", file=sys.stderr)
                n_bad += 1
                continue
            fh.write(json.dumps(finalize(obj, sig, tid, a.model)) + "\n")
            n_ok += 1
    print(f"{n_ok} valid, {n_bad} invalid -> {a.out}")


def cmd_report(a):
    from collections import Counter
    recs = [json.loads(l) for l in open(a.results)]
    n = len(recs)
    cat, prim, comp, fr, ft, out = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    flagged = lo = hi = 0
    for r in recs:
        s = r["assessment"]
        cat.update(s["task"]["categories"])
        prim[s["task"]["primary"]] += 1
        comp.update(i["category"] for i in s["compliance"]["issues"])
        fr[s["friction"]["level"]] += 1
        ft.update(e["type"] for e in s["friction"]["events"])
        out[s["success"]["outcome"]] += 1
        flagged += r["derived"]["flag_for_review"]
        lo += r["derived"]["hours_saved_low"] or 0
        hi += r["derived"]["hours_saved_high"] or 0

    def tbl(title, c):
        print(f"\n{title}")
        for k, v in c.most_common():
            print(f"  {k:32s} {v:4d}  {100 * v / n:5.1f}%")
    print(f"{n} transcripts; {flagged} flagged for review; hours saved {lo:.1f}–{hi:.1f}")
    tbl("primary task", prim)
    tbl("task labels (multi)", cat)
    tbl("compliance issues", comp)
    tbl("friction level", fr)
    tbl("friction types", ft)
    tbl("outcome", out)


def cmd_compare(a):
    """Agreement of a candidate run against a reference run on the same transcripts."""
    def load(p):
        return {r["id"]: r for r in map(json.loads, open(p))}
    cand, ref = load(a.candidate), load(a.reference)
    ids = sorted(set(cand) & set(ref))
    if not ids:
        print("no overlapping ids")
        return
    fric = ["none", "low", "moderate", "high"]
    rows = []
    for i in ids:
        c, r = cand[i]["assessment"], ref[i]["assessment"]
        cl, rl = set(c["task"]["categories"]), set(r["task"]["categories"])
        cd, rd = cand[i]["derived"], ref[i]["derived"]
        rows.append(dict(
            id=i,
            primary=c["task"]["primary"] == r["task"]["primary"],
            labels_jaccard=len(cl & rl) / len(cl | rl) if cl | rl else 1.0,
            comp_sev=c["compliance"]["max_severity"] == r["compliance"]["max_severity"],
            comp_flag=cd["flag_for_review"] == rd["flag_for_review"],
            comp_cats={x["category"] for x in c["compliance"]["issues"]}
            == {x["category"] for x in r["compliance"]["issues"]},
            fric_exact=c["friction"]["level"] == r["friction"]["level"],
            fric_within1=abs(fric.index(c["friction"]["level"]) - fric.index(r["friction"]["level"])) <= 1,
            outcome=c["success"]["outcome"] == r["success"]["outcome"],
            hours_c=(cd["manual_hours_low"] + cd["manual_hours_high"]) / 2,
            hours_r=(rd["manual_hours_low"] + rd["manual_hours_high"]) / 2,
            sev_c=c["compliance"]["max_severity"], sev_r=r["compliance"]["max_severity"],
        ))
    n = len(rows)
    print(f"{n} transcripts compared")
    for k in ("primary", "labels_jaccard", "comp_sev", "comp_flag", "comp_cats",
              "fric_exact", "fric_within1", "outcome"):
        print(f"  {k:16s} {sum(r[k] for r in rows) / n:.2f}")

    def rank(v):
        rk = [0.0] * len(v)
        for pos, j in enumerate(sorted(range(len(v)), key=lambda j: v[j])):
            rk[j] = pos
        return rk
    rc, rr = rank([r["hours_c"] for r in rows]), rank([r["hours_r"] for r in rows])
    m = (n - 1) / 2
    num = sum((x - m) * (y - m) for x, y in zip(rc, rr))
    den = (sum((x - m) ** 2 for x in rc) * sum((y - m) ** 2 for y in rr)) ** 0.5
    print(f"  hours_spearman   {num / den:.2f}" if den else "  hours_spearman   None")
    print(f"  hours_total      candidate {sum(r['hours_c'] for r in rows):.1f}  "
          f"reference {sum(r['hours_r'] for r in rows):.1f}")
    print("\nid | primary | severity cand/ref | friction | outcome | manual hours cand/ref")
    for r in rows:
        print(f"  {r['id'][17:58]:41s} {'✓' if r['primary'] else '✗'}  {r['sev_c']:>6s}/{r['sev_r']:<6s} "
              f"{'✓' if r['fric_exact'] else ('~' if r['fric_within1'] else '✗')}  "
              f"{'✓' if r['outcome'] else '✗'}  {r['hours_c']:5.1f}/{r['hours_r']:5.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("digest"); p.add_argument("src", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--keep-empty", action="store_true"); p.set_defaults(f=cmd_digest)
    p = sp.add_parser("emit-prompts"); p.add_argument("digests"); p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_emit)
    p = sp.add_parser("batch-submit"); p.add_argument("digests"); p.add_argument("--model", default=MODEL)
    p.set_defaults(f=cmd_batch_submit)
    p = sp.add_parser("batch-collect"); p.add_argument("batch_id"); p.add_argument("--digests", required=True)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_batch_collect)
    p = sp.add_parser("merge"); p.add_argument("raw"); p.add_argument("--digests", required=True)
    p.add_argument("--out", required=True); p.add_argument("--model", default=MODEL); p.set_defaults(f=cmd_merge)
    p = sp.add_parser("report"); p.add_argument("results"); p.set_defaults(f=cmd_report)
    p = sp.add_parser("compare"); p.add_argument("candidate"); p.add_argument("reference")
    p.set_defaults(f=cmd_compare)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
