// EXPECT-EXIT: 0
// Storing a _Float128 writes all 16 bytes: the padding after the 80-bit image is zero,
// so two stores of the same value compare equal through a union word view.
union u { _Float128 f; unsigned long l[2]; };
__attribute__((noinline)) void put(union u *p, unsigned long lo) {
    p->f = __builtin_nanf128("");
    p->l[0] = lo;
}
int main(void) {
    union u a, b;
    put(&a, 1000);
    put(&b, 1000);
    if (a.l[0] != b.l[0]) return 1;
    if (a.l[1] != b.l[1]) return 2;
    return 0;
}
