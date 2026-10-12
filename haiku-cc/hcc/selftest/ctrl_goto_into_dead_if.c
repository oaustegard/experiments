// EXPECT-EXIT: 0
// Constant-false `if` whose then-branch holds a goto target: falling out of the
// dead branch must not run the else branch.
extern void abort(void);
void test(int *ptr) {
    int i = 1;
    goto useless;
    if (0) {
    useless:
        i = 0;
    } else
        i = 1;
    *ptr = i;
}
int main(void) {
    int i = 1;
    test(&i);
    if (i) abort();
    return 0;
}
