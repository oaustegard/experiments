"""Block-scope `extern` redeclaration of a file-scope tentative array.

A file-scope `T x[];` is a tentative definition whose implicit initializer
gives it one element. A block-scope `extern T x[N];` that completes it to any
other size conflicts with that implicit initialization (GCC: "type of array
'x' completed incompatibly with implicit initialization").
"""

from __future__ import annotations

from hcc import typesys as T


class ExternTentativeMixin:

    def check_extern_completes_tentative(self, prior, ty, name) -> None:
        var = getattr(prior, "var", None) if prior is not None else None
        if var is None or getattr(var, "is_local", False) or getattr(var, "is_extern", False):
            return
        if not getattr(var, "is_tentative", False) or var.init_data is not None:
            return
        if var.ty.kind != T.TypeKind.ARRAY or var.ty.array_len >= 0:
            return
        if ty.kind != T.TypeKind.ARRAY or ty.array_len < 0 or ty.array_len == 1:
            return
        self.error(f"type of array '{name.text}' completed incompatibly with implicit initialization",
                   name.loc)
