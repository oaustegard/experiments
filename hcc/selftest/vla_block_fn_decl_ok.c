// EXPECT-EXIT: 3
// Block-scope function declarations with ordinary returns, and a VLA-returning
// pointer-to-function parameter, remain valid.
int h(void);
int main(void) {
    int n = 3;
    int (*fp)(void) = h;
    extern int *k(void);
    int (*(*pp)(void))[n];
    (void)pp;
    return n;
}
int h(void) { return 0; }
