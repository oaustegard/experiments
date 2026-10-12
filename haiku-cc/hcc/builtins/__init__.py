"""GCC builtins, one family per module (hcc/builtins/<family>.py).

A module registers handlers with three decorators; nothing else lists builtins,
so adding one is adding a file (or a function in its family's file):

    from hcc.builtins import builtin_parser, builtin_codegen, builtin_fold

    @builtin_parser("__builtin_bswap16")      # full spelling, as written in C
    def _parse_bswap16(self, tok):            # self = Parser, positioned after '('
        ...                                   # parse the arguments, return an Expr
                                              # (often self._call(tok, "bswap16", args, ty))

    @builtin_codegen("bswap16")               # nodes.BuiltinCall.name
    def _gen_bswap16(self, node):             # self = CodeGen; result in the accumulator
        ...

    @builtin_fold("bswap16")                  # optional: integer constant folding
    def _fold_bswap16(node, num):             # num(expr) -> int, raises NotConstant
        return ...                            # value (wrapped to node.ty by the caller)

The parser calls the handler between the builtin's '(' and ')'. Handlers are
written like methods: `self` is the Parser / CodeGen, so the shared helpers
(`self._args(n)`, `self._call(...)`, `self.gen_expr`, `self.emit`, ...) work.
A name registered twice raises at the first lookup, naming both places.
Older families whose codegen shares machinery with the rest of codegen keep
`builtin_<name>` methods on a codegen mixin (atomics.py, varargs.py,
overflow.py, fpclassify.py); the registry is consulted first.
"""
from __future__ import annotations

from hcc.registry import Registry

builtin_parser = Registry("parser builtin", "hcc.builtins")
builtin_codegen = Registry("codegen builtin", "hcc.builtins")
builtin_fold = Registry("constant-folded builtin", "hcc.builtins")
