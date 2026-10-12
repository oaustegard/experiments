// EXPECT-EXIT: 0
// A non-constant "i" operand inside a statically dead branch is not an error.
int main(void) {
    unsigned a = 1;
    if (__builtin_constant_p(a))
        asm volatile("# constant %0" :: "i"(a));
    return 0;
}
