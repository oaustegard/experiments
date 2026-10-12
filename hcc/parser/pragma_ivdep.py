"""#pragma GCC ivdep: the preprocessor forwards it as a marker token that
precedes the next statement. The parser sets `parser.ivdep_pending` just before
a `for` statement; `_stmt_for` takes it, and an ivdep loop without a condition
is an error (GCC: "missing loop condition in loop with 'GCC ivdep' pragma")."""
from __future__ import annotations

from hcc.preprocess.directives import PRAGMA_IVDEP_MARK


def at_ivdep_marker(parser) -> bool:
    return parser.at(PRAGMA_IVDEP_MARK)


def consume_ivdep_marker(parser) -> None:
    parser.next()
    if parser.at("for"):
        parser.ivdep_pending = True


def take_ivdep(parser) -> bool:
    flag = getattr(parser, "ivdep_pending", False)
    parser.ivdep_pending = False
    return flag
