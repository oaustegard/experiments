"""GNU builtin type names: __builtin_va_list, __float128, __float80, __uint128_t, __auto_type."""
from __future__ import annotations

from hcc import typesys as T
from hcc.parser.typespecs import type_specifier


@type_specifier("__builtin_va_list")
def _va_list(self, tok, attr):
    return T.va_list_type()


@type_specifier("__float128")
def _float128(self, tok, attr):
    return T.TY_FLOAT128


@type_specifier("__uint128_t")
def _uint128(self, tok, attr):
    return T.TY_UINT128


@type_specifier("__auto_type")
def _auto_type(self, tok, attr):
    return None       # resolved by the declaration's initializer


@type_specifier("__float80")
def _float80(self, tok, attr):
    return T.TY_LDOUBLE       # x87 extended precision: long double on x86-64
