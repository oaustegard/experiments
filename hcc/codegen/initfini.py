"""GNU constructor/destructor functions: pointer entries in .init_array/.fini_array.

    InitFiniMixin.emit_init_fini()   called once after all objects are emitted.

`__attribute__((constructor[(N)]))` puts the function's address in
`.init_array[.N]`; `destructor` uses `.fini_array[.N]`. The linker sorts the
numbered sections by priority, matching GCC's ordering.
"""
from __future__ import annotations

from hcc import nodes as N

_SECTIONS = {"init": ".init_array", "fini": ".fini_array"}


class InitFiniMixin:

    def emit_init_fini(self) -> None:
        for obj in self.tu.objs:
            if not isinstance(obj, N.Function) or not obj.is_definition or not obj.init_kind:
                continue
            base = _SECTIONS[obj.init_kind]
            name = f"{base}.{obj.init_prio}" if obj.init_prio else base
            self.directive(f'.section {name},"aw",@init_array' if obj.init_kind == "init"
                           else f'.section {name},"aw",@fini_array')
            self.directive(f".quad {obj.symbol}")
