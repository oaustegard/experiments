// EXPECT-EXIT: 0
// scalar_storage_order("big-endian") bitfields: byte image is big-endian.
struct S { short i : 12; char c1 : 1; char c2 : 1; char c3 : 1; char c4 : 1; }
    __attribute__((scalar_storage_order("big-endian")));
struct T { int i : 24; char c1 : 1; char c2 : 1; char c3 : 1; char c4 : 1; }
    __attribute__((scalar_storage_order("big-endian")));
void abort(void);
int main(void) {
    struct S s = { 341, 1, 1, 1, 1 };
    struct T t = { 1193046, 1, 1, 1, 1 };
    unsigned char *p = (unsigned char *)&s, *q = (unsigned char *)&t;
    if (p[0] != 21 || p[1] != 95) abort();
    if (q[0] != 18 || q[3] != 240) abort();
    t.c2 = 0;
    if (t.i != 1193046 || t.c1 != -1 || t.c2 != 0 || t.c4 != -1) abort();
    s.i = -5;
    if (s.i != -5 || s.c4 != -1) abort();
    return 0;
}
