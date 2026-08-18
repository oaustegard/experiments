"""Zero-editorial schemas for the same 18 tools, derived by introspection.

The control arm. `needle.tool` already builds a JSON schema from a function's
signature and its Google-style docstring, and `browsing-bluesky/scripts/bsky.py`
is written that way, so wiring the skill to Needle costs one decorator per
function and no thought at all. That is the arm: what a developer gets for free.

Everything the tuned arm edits is left alone here — the summary line stays as
written for a human reader, `transcribe` and `stopwords` stay in the signature,
`depth` and `parent_height` stay declared, and no numeric bounds are added.

The two wrappers in `catalogue.py` (`resolve_identity`, `atproto_status`) are
decorated as-is for the same reason.
"""

from __future__ import annotations

import needle

from . import catalogue

# Names that are cheap enough to route but whose bsky.py signature is not
# annotated the way `needle.tool` wants. None so far; kept as the seam.
_SKIP: set[str] = set()


def _schema_for(name: str, fn) -> dict:
    decorated = needle.tool(fn)
    return decorated._needle_tool


def build() -> list[dict]:
    ex = catalogue.executors()
    out = []
    for name in catalogue.TOOL_NAMES:
        if name in _SKIP:
            continue
        schema = dict(_schema_for(name, ex[name]))
        # bsky.py's callables keep their own __name__; the two atproto wrappers
        # already match. Pin the declared name to the catalogue name so the two
        # arms are comparable by name.
        schema["name"] = name
        out.append(schema)
    return out


SCHEMAS = None  # built lazily by build(); importing needle is not free
