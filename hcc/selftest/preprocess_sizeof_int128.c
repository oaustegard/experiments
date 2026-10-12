// EXPECT-EXIT: 0
// __SIZEOF_INT128__ is advertised (__int128 has full arithmetic in codegen/int128.py),
// so `__SIZEOF_INT128__ * __CHAR_BIT__` must be usable in #if and in constant expressions.
#ifndef __SIZEOF_INT128__
#error __SIZEOF_INT128__ must be defined
#endif
#if __SIZEOF_INT128__ != 16
#error __SIZEOF_INT128__ must be 16
#endif
_Static_assert (__SIZEOF_INT128__ * __CHAR_BIT__ == 128, "int128 width");
int main(void) {
    return 0;
}
