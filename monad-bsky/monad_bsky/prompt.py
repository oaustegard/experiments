"""Prompt format for Monad over the Bluesky tool catalogue.

Monad's tokenizer has 8,192 tokens and was trained on SYNTH, which is English
prose. Feeding it the JSON schemas Needle consumes costs **1,972 tokens** for
the 18-tool catalogue against a 2,048-token context — there is no room left for
the query, the thinking trace and the answer. The same information rendered as
one prose line per tool costs **574**. That is the whole reason this module
exists, and it is why Monad sees all 18 tools at once while Needle sees the five
its retrieval head selects.

Prompt shape follows the model card: Qwen-style `<|im_start|>` turns and a
`<think>` trace, single-turn only (the card says multi-turn is unsupported).
None of those markers are special tokens in this tokenizer — `<|im_start|>`
encodes as five ordinary pieces — so the template is plain text and the real
specials are `<|begin_of_text|>` (1), `<|end_of_text|>` (2) and `[PAD]` (3).
"""

from __future__ import annotations

import json
import re
from typing import Any

INSTRUCTION = (
    "You call one tool to answer the request. Reply with one JSON object: "
    '{"name": <tool name>, "arguments": {<argument>: <value>}}. '
    "Use only arguments whose values appear in the request. "
    'If no tool fits the request, reply {"name": null, "arguments": {}}.'
)


def tool_line(schema: dict) -> str:
    """One tool as a prose line: `name(args): description`."""
    props = schema.get("parameters", {}).get("properties", {})
    return f"{schema['name']}({', '.join(props)}): {schema['description']}"


def render_tools(schemas: list[dict]) -> str:
    return "\n".join(tool_line(s) for s in schemas)


def build_prompt(schemas: list[dict], query: str) -> str:
    """The full user turn plus the assistant opener the model completes."""
    return (
        "<|im_start|>user\n"
        f"{INSTRUCTION}\n\n"
        f"Tools:\n{render_tools(schemas)}\n\n"
        f"Request: {query}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def build_target(reasoning: str | None, name: str | None, arguments: dict | None) -> str:
    """The assistant completion: a short trace, then exactly one JSON object."""
    call = {"name": name, "arguments": arguments or {}}
    trace = (reasoning or "no tool fits").strip()
    return f"{trace}\n</think>\n{json.dumps(call)}<|im_end|>"


_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_call(text: str) -> tuple[str | None, dict, str]:
    """Pull the call out of a completion.

    Returns `(name, arguments, status)` where status is one of `ok`,
    `refused` (a well-formed call naming no tool), or `unparseable`.
    Monad decodes unconstrained — unlike Needle, whose grammar makes malformed
    JSON impossible — so the parse rate is itself a measurement.
    """
    body = text.split("</think>", 1)[-1]
    body = body.split("<|im_end|>", 1)[0].strip()
    m = _OBJ.search(body)
    if not m:
        return None, {}, "unparseable"
    try:
        obj: Any = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, {}, "unparseable"
    if not isinstance(obj, dict):
        return None, {}, "unparseable"
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(args, dict):
        args = {}
    if name in (None, "", "null"):
        return None, {}, "refused"
    return str(name), args, "ok"


def thinking(text: str) -> str:
    return text.split("</think>", 1)[0].strip()
