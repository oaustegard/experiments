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
    is_inline: bool = False
    is_tls: bool = False
    is_noreturn: bool = False
    align: int = 0
    is_weak: bool = False
    is_packed: bool = False
    reverse_sso: bool = False       # scalar_storage_order("big-endian")
    section: str | None = None
    visibility: str | None = None
    alias: str | None = None
    asm_name: str | None = None
    vector_size: int = 0
    mode: str | None = None
