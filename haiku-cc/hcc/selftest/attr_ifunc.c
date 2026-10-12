// EXPECT-EXIT: 0
static int impl(void) { return 0; }
static __typeof__(impl) *resolver(void) { return impl; }
static int magic(void) __attribute__((ifunc("resolver")));
int main(void) { return magic() != 0; }
