"""Parsing of special __builtin_* forms (those taking types, or needing
compile-time evaluation). Plain library builtins such as __builtin_memcpy
are handled in primary.py by calling the libc function of the same name.

BuiltinParseMixin.parse_builtin(name_tok) -> Expr
The handlers live in hcc/builtins/<family>.py, registered with
`@builtin_parser("__builtin_<name>")` (see hcc/builtins/__init__.py); nothing
here lists them. This mixin keeps the dispatch and the shared helpers
`_args` / `_call` that handlers use.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.builtins import builtin_parser
from hcc.builtins.floatconst import floatn_libc_name  # noqa: F401  (primary.py)

# Every registered spelling (a lazily loaded registry; supports `in`).
SPECIAL_BUILTINS = builtin_parser


class BuiltinParseMixin:

    def parse_builtin(self, tok):
        handler = builtin_parser.get(tok.text)
        self.expect("(")
        node = handler(self, tok)
        self.expect(")")
        return node

    def _args(self, n=None):
        args = []
        while not self.at(")"):
            if args:
                self.expect(",")
            args.append(self.typed(self.assign()))
        if n is not None and len(args) != n:
            self.error(f"builtin expects {n} arguments")
        return args

    def _call(self, tok, name, args, ty, type_arg=None):
        return N.BuiltinCall(tok.loc, name, args, type_arg, ty=ty)
