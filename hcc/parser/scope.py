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
    unavailable: bool = False       # typedef declared with __attribute__((unavailable))


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
    is_auto: bool = False           # `auto` storage class (block-scope nested function decls)
    is_inline: bool = False
    is_gnu_inline: bool = False     # __attribute__((gnu_inline)): gnu89 inline semantics
    is_tls: bool = False
    is_noreturn: bool = False
    is_constexpr: bool = False      # C23 `constexpr`: the object itself is const (see declspec)
    align: int = 0
    alignas_min: int = 0            # largest _Alignas(N) on the declaration (may not be lowered)
    is_weak: bool = False
    is_weakref: bool = False        # __attribute__((weakref("t"))) on a declaration
    is_unavailable: bool = False    # __attribute__((unavailable)) on the declaration
    unavail_msg: str | None = None  # the unavailable("msg") text, if any
    fn_attrs: set = field(default_factory=set)  # leaf, returns_nonnull, ... (names only)
    is_nocommon: bool = False       # __attribute__((nocommon)): tentative def goes to .bss
    warn_if_not_aligned: bool = False   # type-only attribute: rejected on objects/functions/fields
    is_packed: bool = False
    is_designated_init: bool = False  # __attribute__((designated_init)): struct-only (checked)
    nonnull_noargs: bool = False    # bare __attribute__((nonnull)): invalid on a non-prototype
    is_ms_struct: bool = False
    reverse_sso: bool = False       # scalar_storage_order("big-endian")
    section: str | None = None
    strict_flex_array: bool = False  # __attribute__((strict_flex_array(N))) on a field
    visibility: str | None = None
    alias: str | None = None
    ifunc: str | None = None        # __attribute__((ifunc("resolver")))
    init_kind: str | None = None    # "init" for constructor, "fini" for destructor
    init_prio: int | None = None    # constructor(N) / destructor(N) priority (None: default)
    cleanup: str | None = None      # cleanup(f) on an automatic variable: handler name f
    asm_name: str | None = None
    vector_size: int = 0
    mode: str | None = None
