"""__atomic_* / __sync_* builtins: parsed by parser/atomic_builtins.py, lowered by
codegen/atomics.py (builtin_atomic_* / builtin_sync_* methods)."""
from __future__ import annotations

from hcc.builtins import builtin_fold, builtin_parser
from hcc.parser.atomic_builtins import ATOMIC_BUILTINS, atomic_call

# operand sizes that x86-64 performs lock-free (matches codegen/atomics.py _SUF)
_ALWAYS_LOCK_FREE_SIZES = frozenset([1, 2, 4, 8])


@builtin_parser(*sorted(ATOMIC_BUILTINS))
def _parse_atomic(self, tok):
    return atomic_call(self, tok)


@builtin_fold("atomic_always_lock_free")
def _fold_always_lock_free(node, num):
    size = num(node.args[0])     # size is constant; ptr only feeds alignment (ignored)
    return int(size in _ALWAYS_LOCK_FREE_SIZES)
