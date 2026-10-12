// EXPECT-EXIT: 0
// __atomic_test_and_set / __atomic_clear / __atomic_is_lock_free (atomic_flag ops).
extern void abort(void);
static unsigned char flag;
int main(void) {
    if (__atomic_is_lock_free(sizeof(int), &flag) != 1) abort();
    if (__atomic_is_lock_free(16, &flag) != 0) abort();
    if (__atomic_test_and_set(&flag, __ATOMIC_SEQ_CST) != 0) abort();
    if (__atomic_test_and_set(&flag, __ATOMIC_ACQ_REL) != 1) abort();
    __atomic_clear(&flag, __ATOMIC_SEQ_CST);
    if (flag != 0) abort();
    if (__atomic_test_and_set(&flag, __ATOMIC_RELAXED) != 0) abort();
    return 0;
}
