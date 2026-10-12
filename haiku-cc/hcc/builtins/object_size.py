"""__builtin_object_size / __builtin_dynamic_object_size (folded by parser/object_size.py)."""
from __future__ import annotations

from hcc.builtins import builtin_parser
from hcc.parser.object_size import lower_object_size


@builtin_parser("__builtin_object_size", "__builtin_dynamic_object_size")
def _parse_object_size(self, tok):
    return lower_object_size(self, tok)
