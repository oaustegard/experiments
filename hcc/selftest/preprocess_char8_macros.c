// EXPECT-EXIT: 0
// __CHAR8_TYPE__ names the C23 char8_t representation; the atomic lock-free
// macro for it is predefined as gcc does.
#if !defined(__GCC_ATOMIC_CHAR8_T_LOCK_FREE)
#error missing lock-free macro
#endif
_Atomic __CHAR8_TYPE__ ac8;
int main(void) {
    __CHAR8_TYPE__ c = 255;
    return (sizeof(__CHAR8_TYPE__) == 1 && c == 255 && __GCC_ATOMIC_CHAR8_T_LOCK_FREE == 2) ? 0 : 1;
}
