"""Parsing of the __atomic_* builtins (the ones gcc's <stdatomic.h> expands to).

ATOMIC_BUILTINS is registered with the builtin parser by hcc/builtins/atomic.py.
atomic_call(parser, tok) -> BuiltinCall; the node name is the builtin name
without its `__` prefix (e.g. "atomic_fetch_add"); codegen/atomics.py lowers it.
Operand sizes are taken from the pointer argument in codegen.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser.addr_operand import is_rvalue_operand

_RMW_OPS = ("add", "sub", "and", "or", "xor", "nand")

# Arithmetic/bitwise read-modify-write builtins (both __atomic_ and __sync_ spellings).
_RMW_ARITH_NAMES = frozenset(
    [f"__atomic_fetch_{o}" for o in _RMW_OPS] + [f"__atomic_{o}_fetch" for o in _RMW_OPS]
    + [f"__sync_fetch_and_{o}" for o in _RMW_OPS] + [f"__sync_{o}_and_fetch" for o in _RMW_OPS])
# Builtins returning the pointee type (value-producing), the bool/int result
# of compare_exchange, and void for the rest.
_VALUE_RET = frozenset(
    [f"__atomic_fetch_{o}" for o in _RMW_OPS] + [f"__atomic_{o}_fetch" for o in _RMW_OPS]
    + ["__atomic_exchange_n", "__atomic_load_n"])
_INT_RET = frozenset(["__atomic_compare_exchange_n", "__atomic_compare_exchange"])
# atomic_flag ops: test_and_set yields _Bool, clear yields void.
_FLAG_BOOL_RET = frozenset(["__atomic_test_and_set"])
_VOID_RET = frozenset(["__atomic_store_n", "__atomic_load", "__atomic_store", "__atomic_exchange",
                       "__atomic_thread_fence", "__atomic_signal_fence", "__atomic_clear"])
# __atomic_is_lock_free(size, ptr): the first argument is a size, not a pointer.
_LOCK_FREE = "__atomic_is_lock_free"
# __atomic_always_lock_free(size, ptr): same argument shape, folded to a constant.
_ALWAYS_LOCK_FREE = "__atomic_always_lock_free"
_LOCK_FREE_NAMES = frozenset([_LOCK_FREE, _ALWAYS_LOCK_FREE])
# Generic (size-less) __atomic_* forms: every pointer argument points at a data object.
_GENERIC_NAMES = frozenset(["__atomic_load", "__atomic_store", "__atomic_exchange",
                            "__atomic_compare_exchange"])
# Legacy gcc __sync_* builtins share the same lowering (full barriers on x86).
_SYNC_VALUE_RET = frozenset(
    [f"__sync_fetch_and_{o}" for o in _RMW_OPS] + [f"__sync_{o}_and_fetch" for o in _RMW_OPS]
    + ["__sync_lock_test_and_set", "__sync_val_compare_and_swap"])
_SYNC_INT_RET = frozenset(["__sync_bool_compare_and_swap"])
_SYNC_VOID_RET = frozenset(["__sync_lock_release", "__sync_synchronize"])
SYNC_BUILTINS = _SYNC_VALUE_RET | _SYNC_INT_RET | _SYNC_VOID_RET
ATOMIC_BUILTINS = (_VALUE_RET | _INT_RET | _VOID_RET | _FLAG_BOOL_RET | SYNC_BUILTINS
                   | _LOCK_FREE_NAMES)


# Generic (size-taking) forms: index of each pointer operand after the first.
_GENERIC_PTR_ARGS = {
    "__atomic_load": (1,), "__atomic_store": (1,), "__atomic_exchange": (1, 2),
    "__atomic_compare_exchange": (1, 2),
}


# Exact argument counts (gcc: "too few/too many arguments to function").
_ARITY = {
    "__atomic_load_n": 2, "__atomic_store_n": 3, "__atomic_exchange_n": 3,
    "__atomic_compare_exchange_n": 6, "__atomic_load": 3, "__atomic_store": 3,
    "__atomic_exchange": 4, "__atomic_compare_exchange": 6,
    "__atomic_test_and_set": 2, "__atomic_clear": 2,
    "__atomic_thread_fence": 1, "__atomic_signal_fence": 1,
    "__atomic_is_lock_free": 2, "__atomic_always_lock_free": 2,
    "__sync_lock_release": 1, "__sync_synchronize": 0,
    "__sync_lock_test_and_set": 2, "__sync_val_compare_and_swap": 3,
    "__sync_bool_compare_and_swap": 3,
}
for _o in _RMW_OPS:
    _ARITY[f"__atomic_fetch_{_o}"] = 3
    _ARITY[f"__atomic_{_o}_fetch"] = 3
    _ARITY[f"__sync_fetch_and_{_o}"] = 2
    _ARITY[f"__sync_{_o}_and_fetch"] = 2


def _check_arity(name, args, tok, parser):
    want = _ARITY.get(name)
    if want is None or len(args) == want:
        return
    which = "few" if len(args) < want else "many"
    parser.error(f"too {which} arguments to function '{name}'", tok)


def _check_operand_pointers(parser, name, args, tok):
    """gcc: argument 1 must be a non-void pointer; the other pointer operands of the
    generic __atomic_* forms must be pointers."""
    pointee = args[0].ty.base
    if pointee is None or pointee.unqual().kind is T.TypeKind.VOID or _rvalue_array(args[0]):
        parser.error(f"argument 1 of '{name}' must be a non-void pointer type", tok)
    for idx in _GENERIC_PTR_ARGS.get(name, ()):
        if idx < len(args) and (not T.is_pointer_like(args[idx].ty) or _rvalue_array(args[idx])):
            parser.error(f"argument {idx + 1} of '{name}' must be a pointer type", tok)


def _rvalue_array(node):
    """An array that is not an lvalue (e.g. a member of a call result) does not decay."""
    return node.ty.kind is T.TypeKind.ARRAY and is_rvalue_operand(node)


def atomic_call(parser, tok):
    """Parse the argument list (the '(' is already consumed) of an __atomic_* call."""
    name = tok.text
    args = parser._args()
    _check_arity(name, args, tok, parser)
    if name in ("__atomic_thread_fence", "__atomic_signal_fence"):
        ty = T.TY_VOID
    elif name == "__sync_synchronize":
        ty = T.TY_VOID
    elif name in _LOCK_FREE_NAMES:
        ty = T.TY_BOOL
    elif name in _FLAG_BOOL_RET:
        if not args or not T.is_pointer_like(args[0].ty):
            parser.error(f"'{name}' expects a pointer as first argument", tok)
        ty = T.TY_BOOL
    else:
        if not args or not T.is_pointer_like(args[0].ty):
            parser.error(f"'{name}' expects a pointer as first argument", tok)
        if name in _GENERIC_NAMES:
            # gcc: the generic (size-less) forms take pointers to data objects; a pointer
            # to a function in any pointer slot after the first is rejected.
            for i, a in enumerate(args[1:], start=2):
                if (T.is_pointer_like(a.ty) and a.ty.base.kind is T.TypeKind.FUNC):
                    parser.error(f"argument {i} of '{name}' must not be a pointer to a function",
                                 tok)
        if name in _RMW_ARITH_NAMES and args[0].ty.base.unqual().kind is T.TypeKind.BOOL:
            # gcc (PR 68966): arithmetic/bitwise RMW on _Bool would break its representation.
            parser.error(f"operand type '{args[0].ty.base}' is incompatible with argument 1 "
                         f"of '{name}'", tok)
        _check_operand_pointers(parser, name, args, tok)
        if name in _VALUE_RET or name in _SYNC_VALUE_RET:
            ty = args[0].ty.base.unqual()
        elif name in _INT_RET or name in _SYNC_INT_RET:
            ty = T.TY_BOOL     # gcc: compare_exchange / bool_compare_and_swap yield _Bool
        else:
            ty = T.TY_VOID
    return N.BuiltinCall(tok.loc, name[2:], args, None, ty=ty)
