// EXPECT-EXIT: 0
// Width macros (__SCHAR_WIDTH__ etc.) used by <limits.h> must be predefined.
_Static_assert (__SCHAR_WIDTH__ == 8, "schar width");
_Static_assert (__INT_WIDTH__ == 32, "int width");
_Static_assert (__LONG_WIDTH__ == 64, "long width");
int main (void) { return 0; }
