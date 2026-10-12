// EXPECT-EXIT: 0
// gcc *_WIDTH__ and __GCC_ATOMIC_*_LOCK_FREE predefined macros must exist.
int main(void) {
    int a = __LONG_LONG_WIDTH__ - 1 == 63;
    int b = __SCHAR_WIDTH__ == 8 && __INT_LEAST16_WIDTH__ == 16;
    int c = __GCC_ATOMIC_BOOL_LOCK_FREE == 2 && __GCC_ATOMIC_TEST_AND_SET_TRUEVAL == 1;
    return !(a && b && c);
}
