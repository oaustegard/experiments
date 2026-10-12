// EXPECT-EXIT: 0
// Huge array members in an initializer must not allocate per-element nodes,
// and large aggregate copies must not be unrolled.
struct S { short a; char b[1 << 20]; };
struct Big { char a[300]; int n; };
struct Big g1, g2;
int main(void) {
    struct S s = { 5 };
    static struct S t = { 7 };
    if (s.a != 5 || s.b[12345] != 0 || t.a != 7 || t.b[1000] != 0) return 1;
    struct Big l;
    __builtin_memset(&l, 7, sizeof l);
    l.n = 5;
    g2 = l;
    g1 = g2;
    if (g1.n != 5 || g1.a[299] != 7) return 2;
    return 0;
}
