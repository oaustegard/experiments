"""Lowering of nodes.BuiltinCall (name without the `__builtin_` prefix).

The handlers live in hcc/builtins/<family>.py, registered with
`@builtin_codegen("<name>")` (see hcc/builtins/__init__.py). Families whose
lowering shares codegen machinery keep `builtin_<name>` methods on their mixin
(atomics.py, varargs.py, overflow.py, fpclassify.py); the registry wins.
"""
from __future__ import annotations

from hcc import diag
from hcc.builtins import builtin_codegen


class BuiltinMixin:

    def expr_BuiltinCall(self, node) -> None:
        fn = builtin_codegen.get(node.name)
        if fn is not None:
            fn(self, node)
            return
        fn = getattr(self, "builtin_" + node.name, None)
        if fn is None:
            diag.error(node, f"codegen: unsupported builtin '__builtin_{node.name}'", "codegen")
        fn(node)
