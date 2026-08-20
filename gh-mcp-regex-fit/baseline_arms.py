#!/usr/bin/env python3
"""The arms from the first pass, wrapped in the `arms.py` interface."""

from __future__ import annotations

from pathlib import Path

from arms import register
from handwritten import HandRouter
from router import Router

HERE = Path(__file__).resolve().parent

for _tag in ("schema", "open", "cues", "laplace8", "overlap", "overlap-lap8", "uni-lap8"):
    register(f"fitted-{_tag}", lambda t=_tag: Router(HERE / f"rules_{t}.json"))
register("hand", lambda: HandRouter())
register("hand+fallback", lambda: HandRouter(fallback=True))
