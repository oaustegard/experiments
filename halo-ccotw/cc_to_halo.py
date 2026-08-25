"""Convert Claude Code transcripts into HALO/OpenInference-shaped OTel spans.

Claude Code writes one JSONL record per conversation event under
``~/.claude/projects/<slug>/<session_id>.jsonl``. HALO's engine reads one OTel
span per JSONL line. The mapping is mostly mechanical: the transcript already
carries a parent pointer (``parentUuid``), a session id, per-record timestamps,
and — on assistant records — the model name and a full ``usage`` block.

Two things need care.

**Durations.** Each transcript record has a single ``timestamp``, not a
start/end pair. A span's end_time is its own timestamp; its start_time is the
timestamp of the nearest ancestor record. Tool spans get genuine wall-clock
(call to result). LLM spans get model latency plus whatever the harness spent
between the previous event and the request, which is an upper bound rather than
a measurement.

**Prompt tokens.** Anthropic reports ``input_tokens``,
``cache_read_input_tokens`` and ``cache_creation_input_tokens`` separately.
They are all prompt tokens the model processed, so HALO's
``inference.llm.input_tokens`` gets the sum; counting only ``input_tokens``
reports ~100 for a session that read millions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SCOPE = {"name": "claude-code-transcript", "version": "1"}
AGENT_MAIN = "claude-code"
AGENT_SUB = "claude-code-subagent"


def _iso_nanos(ts: str) -> str:
    """Normalize an ISO-8601 timestamp to the nanosecond form HALO's fixtures use."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}000Z"


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _text_of(content: Any) -> str:
    """Flatten an Anthropic content list (or bare string) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)[:20000]
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            parts.append("[thinking] " + block.get("thinking", ""))
        elif block.get("type") == "tool_use":
            parts.append(
                f"[tool_use {block.get('name')}] "
                + json.dumps(block.get("input", {}), ensure_ascii=False)
            )
        elif block.get("type") == "tool_result":
            parts.append("[tool_result] " + _text_of(block.get("content", "")))
    return "\n".join(parts)


def _span(
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str,
    name: str,
    kind: str,
    start: str,
    end: str,
    status_code: str,
    status_message: str,
    resource_attrs: dict[str, Any],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "trace_state": "",
        "name": name,
        "kind": kind,
        "start_time": start,
        "end_time": end,
        "status": {"code": status_code, "message": status_message},
        "resource": {"attributes": resource_attrs},
        "scope": dict(SCOPE),
        "attributes": attributes,
    }


def _agent_identity(rec: dict[str, Any]) -> dict[str, Any]:
    """Sidechain records are subagent work and get their own agent name.

    HALO rolls up per ``inference.agent_name``; splitting subagents out is what
    lets the analysis separate main-loop turns from delegated ones. Every span
    carries it — the index counts spans without it as missing identity.
    """
    name = AGENT_SUB if rec.get("isSidechain") else AGENT_MAIN
    return {"inference.agent_name": name, "agent.name": name}


def convert(transcript: Path, project_id: str, service_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in transcript.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    convo = [r for r in records if r.get("type") in ("user", "assistant") and "uuid" in r]
    if not convo:
        return []

    # The parent chain walks through every record, not just the conversational
    # ones: attachment, hook, and tool_result-only records all carry uuids and
    # sit mid-chain but never become spans.
    all_by_uuid = {r["uuid"]: r for r in records if "uuid" in r}

    def is_tool_result_only(rec: dict[str, Any]) -> bool:
        msg = rec.get("message")
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        return bool(
            isinstance(content, list)
            and content
            and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        )

    emitted = {
        r["uuid"]
        for r in convo
        if isinstance(r.get("message"), dict)
        and (r.get("type") == "assistant" or not is_tool_result_only(r))
    }

    # tool_use_id -> the user record and block carrying its result
    results_by_tool_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("type") != "user":
            continue
        content = rec.get("message", {}).get("content") if isinstance(rec.get("message"), dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                if tid:
                    results_by_tool_id[tid] = {"record": rec, "block": block}

    session_id = convo[0].get("sessionId", "unknown")
    root_id = f"session-{session_id}"
    first, last = _parse(convo[0]["timestamp"]), _parse(convo[-1]["timestamp"])
    meta = convo[-1]

    resource_attrs = {
        "service.name": service_name,
        "service.version": meta.get("version", ""),
        "deployment.environment": meta.get("entrypoint", "unknown"),
        "claude_code.cwd": meta.get("cwd", ""),
        "claude_code.git_branch": meta.get("gitBranch", ""),
    }

    def resolve_parent(rec: dict[str, Any]) -> str:
        """Nearest emitted ancestor, falling back to the session root span."""
        cur = rec.get("parentUuid")
        seen: set[str] = set()
        while cur and cur in all_by_uuid and cur not in seen:
            seen.add(cur)
            if cur in emitted:
                return cur
            cur = all_by_uuid[cur].get("parentUuid")
        return root_id

    def span_start(rec: dict[str, Any]) -> str:
        """This span's start: the timestamp of the nearest ancestor record."""
        cur = rec.get("parentUuid")
        seen: set[str] = set()
        while cur and cur in all_by_uuid and cur not in seen:
            seen.add(cur)
            parent = all_by_uuid[cur]
            if "timestamp" in parent:
                return _iso_nanos(parent["timestamp"])
            cur = parent.get("parentUuid")
        fallback = _parse(rec["timestamp"]) - timedelta(milliseconds=1)
        return _iso_nanos(fallback.isoformat())

    spans: list[dict[str, Any]] = [
        _span(
            trace_id=session_id,
            span_id=root_id,
            parent_span_id="",
            name="claude-code-session",
            kind="SPAN_KIND_INTERNAL",
            start=_iso_nanos(convo[0]["timestamp"]),
            end=_iso_nanos(convo[-1]["timestamp"]),
            status_code="STATUS_CODE_OK",
            status_message="",
            resource_attrs=resource_attrs,
            attributes={
                "openinference.span.kind": "AGENT",
                "inference.export.schema_version": SCHEMA_VERSION,
                "inference.project_id": project_id,
                "inference.observation_kind": "AGENT",
                "inference.agent_name": AGENT_MAIN,
                "agent.name": AGENT_MAIN,
                "claude_code.session_id": session_id,
                "claude_code.record_count": len(convo),
                "claude_code.sidechain_records": sum(1 for r in convo if r.get("isSidechain")),
                "claude_code.duration_seconds": round((last - first).total_seconds(), 3),
            },
        )
    ]

    for rec in convo:
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        uuid = rec["uuid"]
        parent_span = resolve_parent(rec)
        start, end = span_start(rec), _iso_nanos(rec["timestamp"])

        if rec.get("type") == "assistant":
            usage = msg.get("usage") or {}
            content = msg.get("content") or []
            tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            stop = msg.get("stop_reason") or ""
            is_err = stop in ("refusal", "max_tokens") or bool(msg.get("diagnostics"))
            prompt_tokens = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

            attrs: dict[str, Any] = {
                **_agent_identity(rec),
                "openinference.span.kind": "LLM",
                "inference.export.schema_version": SCHEMA_VERSION,
                "inference.project_id": project_id,
                "inference.observation_kind": "LLM",
                "inference.llm.model_name": msg.get("model", ""),
                "inference.llm.input_tokens": prompt_tokens,
                "inference.llm.output_tokens": usage.get("output_tokens", 0),
                "llm.model_name": msg.get("model", ""),
                "llm.token_count.prompt": prompt_tokens,
                "llm.token_count.completion": usage.get("output_tokens", 0),
                "llm.token_count.prompt_details.uncached": usage.get("input_tokens", 0),
                "llm.token_count.prompt_details.cache_read": usage.get("cache_read_input_tokens", 0),
                "llm.token_count.prompt_details.cache_write": usage.get("cache_creation_input_tokens", 0),
                "output.value": _text_of(content),
                "claude_code.stop_reason": stop,
                "claude_code.request_id": rec.get("requestId", ""),
                "claude_code.effort": rec.get("effort", ""),
                "claude_code.is_sidechain": bool(rec.get("isSidechain")),
                "claude_code.tool_calls": [b.get("name") for b in tool_uses],
            }
            if rec.get("attributionMcpServer"):
                attrs["claude_code.mcp_server"] = rec["attributionMcpServer"]

            spans.append(_span(
                trace_id=session_id, span_id=uuid, parent_span_id=parent_span,
                name=f"llm.{msg.get('model', 'unknown')}", kind="SPAN_KIND_CLIENT",
                start=start, end=end,
                status_code="STATUS_CODE_ERROR" if is_err else "STATUS_CODE_OK",
                status_message=stop if is_err else "",
                resource_attrs=resource_attrs, attributes=attrs,
            ))

            # One TOOL span per tool_use, spanning call -> result.
            for block in tool_uses:
                tid = block.get("id") or ""
                hit = results_by_tool_id.get(tid)
                t_end = _iso_nanos(hit["record"]["timestamp"]) if hit else end
                err = bool(hit and hit["block"].get("is_error"))
                out = _text_of(hit["block"].get("content", "")) if hit else ""
                spans.append(_span(
                    trace_id=session_id, span_id=f"tool-{tid}", parent_span_id=uuid,
                    name=f"tool.{block.get('name', 'unknown')}", kind="SPAN_KIND_INTERNAL",
                    start=end, end=t_end,
                    status_code="STATUS_CODE_ERROR" if err else "STATUS_CODE_OK",
                    status_message="tool_error" if err else "",
                    resource_attrs=resource_attrs,
                    attributes={
                        **_agent_identity(rec),
                        "openinference.span.kind": "TOOL",
                        "inference.export.schema_version": SCHEMA_VERSION,
                        "inference.project_id": project_id,
                        "inference.observation_kind": "TOOL",
                        "tool.name": block.get("name", "unknown"),
                        "input.value": json.dumps(block.get("input", {}), ensure_ascii=False),
                        "output.value": out,
                        "claude_code.tool_use_id": tid,
                        "claude_code.tool_result_missing": hit is None,
                    },
                ))
        else:
            # tool_result-only user turns are already covered by their TOOL span.
            if is_tool_result_only(rec):
                continue
            spans.append(_span(
                trace_id=session_id, span_id=uuid, parent_span_id=parent_span,
                name="user-message", kind="SPAN_KIND_INTERNAL", start=start, end=end,
                status_code="STATUS_CODE_OK", status_message="",
                resource_attrs=resource_attrs,
                attributes={
                    **_agent_identity(rec),
                    "openinference.span.kind": "CHAIN",
                    "inference.export.schema_version": SCHEMA_VERSION,
                    "inference.project_id": project_id,
                    "inference.observation_kind": "CHAIN",
                    "input.value": _text_of(msg.get("content")),
                    "claude_code.prompt_source": rec.get("promptSource", ""),
                    "claude_code.permission_mode": rec.get("permissionMode", ""),
                    "claude_code.is_sidechain": bool(rec.get("isSidechain")),
                },
            ))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Claude Code transcripts to HALO span JSONL.")
    ap.add_argument("transcripts", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--project-id", default="claude-code")
    ap.add_argument("--service-name", default="claude-code-web")
    args = ap.parse_args()

    total = 0
    with args.out.open("w") as fh:
        for path in args.transcripts:
            for span in convert(path, args.project_id, args.service_name):
                fh.write(json.dumps(span, ensure_ascii=False) + "\n")
                total += 1
    print(f"wrote {total} spans from {len(args.transcripts)} transcript(s) -> {args.out}")


if __name__ == "__main__":
    main()
