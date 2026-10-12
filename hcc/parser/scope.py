"""Scope objects for the parser (C has separate ordinary-identifier and tag
namespaces). The Parser keeps a list of Scope, innermost last.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hcc.nodes import Obj
from hcc.typesys import Type


@dataclass
class VarScope:
    """What an ordinary identifier refers to: exactly one field is set."""
    var: Obj | None = None
    type_def: Type | None = None
    enum_ty: Type | None = None
    enum_val: int = 0
    enum_cty: Type | None = None    # C23 type of the enumerator constant itself


@dataclass
class Scope:
    vars: dict[str, VarScope] = field(default_factory=dict)
    tags: dict[str, Type] = field(default_factory=dict)


@dataclass
class VarAttr:
    """Storage class / function specifiers / attributes from declspec."""
    is_typedef: bool = False
    is_static: bool = False
    is_extern: bool = False
    is_register: bool = False
    is_inline: bool = False
    is_gnu_inline: bool = False     # __attribute__((gnu_inline)): gnu89 inline semantics
    is_tls: bool = False
    is_noreturn: bool = False
    align: int = 0
    is_weak: bool = False
    fn_attrs: set = field(default_factory=set)  # leaf, returns_nonnull, ... (names only)
    is_nocommon: bool = False       # __attribute__((nocommon)): tentative def goes to .bss
    is_packed: bool = False
    is_ms_struct: bool = False
    reverse_sso: bool = False       # scalar_storage_order("big-endian")
    section: str | None = None
    visibility: str | None = None
    alias: str | None = None
    ifunc: str | None = None        # __attribute__((ifunc("resolver")))
    init_kind: str | None = None    # "init" for constructor, "fini" for destructor
    init_prio: int | None = None    # constructor(N) / destructor(N) priority (None: default)
    cleanup: str | None = None      # cleanup(f) on an automatic variable: handler name f
    asm_name: str | None = None
    vector_size: int = 0
    mode: str | None = None
