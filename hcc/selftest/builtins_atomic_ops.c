// EXPECT-EXIT: 0
// __atomic_* builtins (as used by <stdatomic.h>) on int, unsigned char and long.
extern void abort(void);
static int g = 5;
static unsigned char c8 = 200;
static long lg = -3;
int main(void) {
    if (__atomic_fetch_add(&g, 7, __ATOMIC_SEQ_CST) != 5 || g != 12) abort();
    __atomic_store_n(&g, 42, __ATOMIC_SEQ_CST);
    if (__atomic_exchange_n(&g, 9, __ATOMIC_SEQ_CST) != 42) abort();
    int e = 9;
    if (!__atomic_compare_exchange_n(&g, &e, 11, 0, 0, 0) || g != 11) abort();
    e = 1;
    if (__atomic_compare_exchange_n(&g, &e, 12, 0, 0, 0) || e != 11) abort();
    if (__atomic_fetch_sub(&c8, 201, 0) != 200 || c8 != 255) abort();
    if (__atomic_add_fetch(&c8, 1, 0) != 0) abort();
    if (__atomic_add_fetch(&lg, 10, 0) != 7) abort();
    if (__atomic_fetch_nand(&lg, 6, 0) != 7 || lg != ~(7L & 6L)) abort();
    return 0;
}
