"""GCC vector builtins: __builtin_shuffle, __builtin_shufflevector, __builtin_convertvector."""
from __future__ import annotations

from hcc.builtins import builtin_codegen, builtin_parser
from hcc.codegen.vector_shuffle import shuffle
from hcc.parser.vector_cast import lower_convertvector
from hcc.parser.vector_shufflevector import lower_shufflevector
from hcc.sema.compat import is_compatible
from hcc import typesys as T


@builtin_parser("__builtin_shufflevector")
def _parse_shufflevector(self, tok):
    return lower_shufflevector(self, tok.loc, self._args())


@builtin_parser("__builtin_convertvector")
def _parse_convertvector(self, tok):
    """__builtin_convertvector(vec, vector_type): element-wise conversion."""
    src = self.typed(self.assign())
    self.expect(",")
    dst = self.typename()
    return lower_convertvector(self, tok.loc, src, dst)


@builtin_parser("__builtin_shuffle")
def _parse_shuffle(self, tok):
    """__builtin_shuffle(v, mask) / __builtin_shuffle(a, b, mask): element
    i of the result is v[mask[i] % N] (or the concatenation a:b for the
    3-argument form, indices taken modulo 2N). Result type is that of the
    first vector."""
    args = self._args()
    if len(args) not in (2, 3):
        self.error("builtin expects 2 or 3 arguments")
    a, mask = args[0], args[-1]
    if not a.ty.is_vector or not mask.ty.is_vector or not T.is_integer(mask.ty.base):
        self.error("'__builtin_shuffle' expects integer vector operands")
    n = a.ty.size // a.ty.base.size
    if mask.ty.size // mask.ty.base.size != n:
        self.error("'__builtin_shuffle' mask must have as many elements as the vector")
    if len(args) == 3 and not is_compatible(args[1].ty, a.ty):
        self.error("'__builtin_shuffle' vector operands must have the same type")
    node = self._call(tok, "shuffle", args, a.ty)
    node.result_buf = self.new_temp(a.ty, tok.loc)
    return node


@builtin_codegen("shuffle")
def _gen_shuffle(self, node) -> None:
    shuffle(self, node)
