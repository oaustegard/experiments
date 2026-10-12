// EXPECT-EXIT: 0
// SIG_ATOMIC_* and WINT_* predefined limits are usable in #if and have gcc values.
#include <stdint.h>
#if SIG_ATOMIC_MIN != -2147483647 - 1 || SIG_ATOMIC_MAX != 2147483647
#error "SIG_ATOMIC limits wrong"
#endif
#if WINT_MIN != 0 || WINT_MAX != 4294967295U
#error "WINT limits wrong"
#endif
int main(void) { return 0; }
