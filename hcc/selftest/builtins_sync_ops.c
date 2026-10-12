// EXPECT-EXIT: 0
// __sync_* legacy builtins: fetch/op-fetch, cas, test-and-set, release.
static char c = 5;
static int i = 10;
int main(void) {
    if (__sync_fetch_and_add(&c, 3) != 5) return 1;
    if (c != 8) return 2;
    if (__sync_add_and_fetch(&i, 2) != 12) return 3;
    if (__sync_bool_compare_and_swap(&i, 12, 40) != 1) return 4;
    if (__sync_val_compare_and_swap(&i, 99, 7) != 40) return 5;
    if (i != 40) return 6;
    if (__sync_lock_test_and_set(&i, 1) != 40) return 7;
    __sync_lock_release(&i);
    if (i != 0) return 8;
    if (__sync_fetch_and_nand(&c, 7) != 8) return 9;
    if (__sync_fetch_and_or(&c, 0) != -1) return 10;
    return 0;
}
