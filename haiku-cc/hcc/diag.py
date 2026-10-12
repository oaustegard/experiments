"""Diagnostics: errors and warnings tagged with a source location and stage.

Usage:
    from hcc import diag
    diag.error(tok, "expected ';'")        # raises CompileError
    diag.warn(tok.loc, "implicit declaration of function 'foo'")
    with diag.stage("parse"):               # driver wraps each stage
        ...

Output format (CONTRACT section 3):  file:line:col: error: [stage] message
"""
from __future__ import annotations

import contextlib
import sys
from typing import NoReturn

from hcc.tokens import Loc, Token


class CompileError(Exception):
    def __init__(self, loc: Loc | None, msg: str, stage: str | None = None):
        super().__init__(msg)
        self.loc = loc
        self.msg = msg
        self.stage = stage

    def format(self) -> str:
        where = f"{self.loc}: " if self.loc else "hcc: "
        return f"{where}error: [{self.stage or 'unknown'}] {self.msg}"


def _loc(where) -> Loc | None:
    if isinstance(where, Token):
        return where.loc
    if isinstance(where, Loc) or where is None:
        return where
    return getattr(where, "loc", None)   # AST nodes


def error(where, msg: str, stage: str | None = None) -> NoReturn:
    """Raise a CompileError at `where` (Token, Loc, AST node or None)."""
    raise CompileError(_loc(where), msg, stage)


def warn(where, msg: str) -> None:
    loc = _loc(where)
    prefix = f"{loc}: " if loc else "hcc: "
    print(f"{prefix}warning: {msg}", file=sys.stderr)


@contextlib.contextmanager
def stage(name: str):
    """Tag any CompileError raised inside the block with stage `name`."""
    try:
        yield
    except CompileError as e:
        if e.stage is None:
            e.stage = name
        raise
