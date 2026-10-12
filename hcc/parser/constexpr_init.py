"""C23 `constexpr` objects: every initializer value must be representable in the
object's element type (gcc: "'constexpr' initializer not representable in type of
object"). Only checked while `Parser.constexpr_init` is set (see initializers.py)."""
from hcc import typesys as T
from hcc.sema.constexpr import try_eval_const

MSG = "'constexpr' initializer not representable in type of object"


def representable(v: int, ty) -> bool:
    bits = ty.size * 8
    if ty.is_unsigned:
        return 0 <= v < (1 << bits)
    return -(1 << (bits - 1)) <= v < (1 << (bits - 1))


MSG_INT = "'constexpr' integer initializer is not an integer constant expression"


def check_scalar(parser, init) -> None:
    """An integer constant leaf `init.expr` converted to the integer type `init.ty`."""
    ty, e = init.ty, init.expr
    if T.is_integer(ty) and (T.is_flonum(e.ty) or e.ty.is_complex):
        # `constexpr int i = 5.0;` / `constexpr int i = 5i;`: the value must be an integer.
        parser.error(MSG_INT, e)
    if ty.kind == T.TypeKind.BOOL or not T.is_integer(ty) or not T.is_integer(e.ty):
        return
    v = try_eval_const(e)
    if v is not None and not representable(v, ty):
        parser.error(MSG, e)


def check_string_units(parser, tok, units, elem) -> None:
    """String literal `tok` whose code units are `units` initializing integer `elem`.
    The source units are `tok`'s element type (u8 literals are unsigned char in C23;
    plain `char` is unsigned under -funsigned-char)."""
    src = tok.ty.base
    src_unsigned = src.is_unsigned or (
        src.kind == T.TypeKind.CHAR and bool(getattr(parser.opts, "char_unsigned", False)))
    bits = src.size * 8
    for v in units:
        if not src_unsigned and v >= 1 << (bits - 1):
            v -= 1 << bits
        if not representable(v, elem):
            parser.error(MSG, tok)
