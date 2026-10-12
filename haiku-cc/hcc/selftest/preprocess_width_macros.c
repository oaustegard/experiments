// EXPECT-EXIT: 0
// C23 width macros (as used by <limits.h>) are predefined.
#ifndef __SCHAR_WIDTH__
# error "missing __SCHAR_WIDTH__"
#endif
_Static_assert(__SCHAR_WIDTH__ == 8, "w");
_Static_assert(__SHRT_WIDTH__ == 16, "w");
int main(void) { return 0; }
