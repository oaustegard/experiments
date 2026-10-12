"""Jumps into the scope of a variably modified identifier (C11 6.8.6.1p1).

A `goto` may not jump forward past the declaration of a variably modified
(VLA, pointer-to-VLA, run-time-sized struct) identifier into the scope that
declaration introduces: the label must lie in the declaring block, after the
declaration, while the goto lies before it.

JumpScopes is owned by one function's parse (Parser.jump_scope, swapped in
and out by function_definition). It records, in source order, the gotos,
labels, VM declarations and the block nesting at each label, and checks them
once the whole body has been parsed.
"""
from __future__ import annotations

from hcc import typesys as T


def is_vm_type(ty, _seen=None) -> bool:
    """True when ty is variably modified: a VLA, a pointer/array/function chain
    ending in one, or a struct with a run-time-sized or variably modified member."""
    seen = _seen if _seen is not None else set()
    while ty is not None and ty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY, T.TypeKind.FUNC):
        ty = ty.return_ty if ty.kind == T.TypeKind.FUNC else ty.base
    if ty is None:
        return False
    if ty.kind == T.TypeKind.VLA:
        return ty.vla_len is not None
    if ty.kind not in (T.TypeKind.STRUCT, T.TypeKind.UNION) or id(ty) in seen:
        return False
    if T.is_dyn_struct(ty):
        return True
    seen.add(id(ty))
    return any(is_vm_type(m.ty, seen) for m in ty.members)


class JumpScopes:

    def __init__(self) -> None:
        self.seq = 0
        self.block_ids: list[int] = []      # enclosing block ids, innermost last
        self.block_seq = 0
        self.se_ids: list[int] = []         # enclosing statement-expression ids
        self.se_seq = 0
        self.gotos: list[tuple] = []        # (seq, name token, enclosing stmt-expr ids)
        self.labels: list[tuple] = []       # (seq, name, enclosing block ids, stmt-expr ids)
        self.vm_decls: list[tuple] = []     # (seq, block id)

    def enter_block(self) -> None:
        self.block_seq += 1
        self.block_ids.append(self.block_seq)

    def leave_block(self) -> None:
        self.block_ids.pop()

    def enter_stmt_expr(self) -> None:
        self.se_seq += 1
        self.se_ids.append(self.se_seq)

    def leave_stmt_expr(self) -> None:
        self.se_ids.pop()

    def _tick(self) -> int:
        self.seq += 1
        return self.seq

    def note_decl(self, ty) -> None:
        if is_vm_type(ty) and self.block_ids:
            self.vm_decls.append((self._tick(), self.block_ids[-1]))

    def note_goto(self, name_tok) -> None:
        self.gotos.append((self._tick(), name_tok, tuple(self.se_ids)))

    def note_label(self, name: str) -> None:
        self.labels.append((self._tick(), name, tuple(self.block_ids), tuple(self.se_ids)))

    def violations(self) -> list:
        """(name token, message) for gotos that jump into a statement expression
        or into the scope of a VM identifier."""
        bad = []
        for gseq, gtok, gse in self.gotos:
            for lseq, lname, lblocks, lse in self.labels:
                if lname != gtok.text:
                    continue
                if any(s not in gse for s in lse):
                    bad.append((gtok, "jump into statement expression"))
                    break
                if lseq < gseq:
                    continue
                if any(gseq < vseq < lseq and vblock in lblocks for vseq, vblock in self.vm_decls):
                    bad.append((gtok, "jump into scope of identifier with variably modified type"))
                    break
        return bad
