"""Element-count limit for GCC vector types (`vector_size` attribute).

GCC rejects a vector whose number of components (vector bytes / element size)
exceeds INT_MAX - 1, e.g. `int __attribute__((vector_size(1ULL << 34)))`.
"""
from __future__ import annotations

from hcc import typesys as T

# GCC: the component count must stay below INT_MAX (LP64 int)
VECTOR_COMPONENTS_MAX = (1 << 31) - 2


def vector_of_checked(parser, base, nbytes: int, where=None):
    """`T.vector_of` with GCC's component-count diagnostic."""
    if base.size and nbytes // base.size > VECTOR_COMPONENTS_MAX:
        parser.error("number of vector components %d exceeds %d"
                     % (nbytes // base.size, VECTOR_COMPONENTS_MAX), where)
    return T.vector_of(base, nbytes)
