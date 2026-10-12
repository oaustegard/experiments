"""Parsing of the __atomic_* builtins (the ones gcc's <stdatomic.h> expands to).

ATOMIC_BUILTINS is merged into SPECIAL_BUILTINS by parser/builtins.py.
atomic_call(parser, tok) -> BuiltinCall; the node name is the builtin name
without its `__` prefix (e.g. "atomic_fetch_add"); codegen/atomics.py lowers it.
Operand sizes are taken from the pointer argument in codegen.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T

_RMW_OPS = ("add", "sub", "and", "or", "xor", "nand")

# Builtins returning the pointee type (value-producing), the bool/int result
# of compare_exchange, and void for the rest.
_VALUE_RET = frozenset(
    [f"__atomic_fetch_{o}" for o in _RMW_OPS] + [f"__atomic_{o}_fetch" for o in _RMW_OPS]
    + ["__atomic_exchange_n", "__atomic_load_n"])
_INT_RET = frozenset(["__atomic_compare_exchange_n", "__atomic_compare_exchange"])
_VOID_RET = frozenset(["__atomic_store_n", "__atomic_load", "__atomic_store", "__atomic_exchange",
                       "__atomic_thread_fence", "__atomic_signal_fence"])
# Legacy gcc __sync_* builtins share the same lowering (full barriers on x86).
_SYNC_VALUE_RET = frozenset(
    [f"__sync_fetch_and_{o}" for o in _RMW_OPS] + [f"__sync_{o}_and_fetch" for o in _RMW_OPS]
    + ["__sync_lock_test_and_set", "__sync_val_compare_and_swap"])
_SYNC_INT_RET = frozenset(["__sync_bool_compare_and_swap"])
_SYNC_VOID_RET = frozenset(["__sync_lock_release", "__sync_synchronize"])
SYNC_BUILTINS = _SYNC_VALUE_RET | _SYNC_INT_RET | _SYNC_VOID_RET
ATOMIC_BUILTINS = _VALUE_RET | _INT_RET | _VOID_RET | SYNC_BUILTINS


def atomic_call(parser, tok):
    """Parse the argument list (the '(' is already consumed) of an __atomic_* call."""
    name = tok.text
    args = parser._args()
    if name in ("__atomic_thread_fence", "__atomic_signal_fence"):
        if len(args) != 1:
            parser.error(f"'{name}' expects 1 argument", tok)
        ty = T.TY_VOID
    elif name == "__sync_synchronize":
        if args:
            parser.error(f"'{name}' expects no arguments", tok)
        ty = T.TY_VOID
    else:
        if not args or not T.is_pointer_like(args[0].ty):
            parser.error(f"'{name}' expects a pointer as first argument", tok)
        if name in _VALUE_RET or name in _SYNC_VALUE_RET:
            ty = args[0].ty.base.unqual()
        elif name in _INT_RET or name in _SYNC_INT_RET:
            ty = T.TY_BOOL     # gcc: compare_exchange / bool_compare_and_swap yield _Bool
        else:
            ty = T.TY_VOID
    return N.BuiltinCall(tok.loc, name[2:], args, None, ty=ty)
