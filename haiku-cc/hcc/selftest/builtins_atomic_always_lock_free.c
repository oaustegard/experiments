// EXPECT-EXIT: 0
// __atomic_always_lock_free folds to a constant for lock-free sizes (usable in _Static_assert).
static int *p;
_Static_assert(__atomic_always_lock_free(1, p), "char is always lock-free");
_Static_assert(__atomic_always_lock_free(8, 0), "8 bytes is always lock-free");
_Static_assert(!__atomic_always_lock_free(16, 0), "16 bytes is not lock-free without cmpxchg16b");
int main(void) {
    return __atomic_always_lock_free(4, p) ? 0 : 1;
}
