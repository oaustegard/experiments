// EXPECT-ERROR: [parse]
// A weak attribute on a block-scope object (no linkage) is an error.
void f(void) {
    __attribute__((weak)) int a;
    (void)a;
}
