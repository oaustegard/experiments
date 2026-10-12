// EXPECT-EXIT: 0
// A scalar operand of a vector binop is converted to the element type and broadcast.
typedef unsigned V __attribute__ ((__vector_size__ (8)));
typedef int VI __attribute__ ((__vector_size__ (8)));
extern void abort(void);
int main(void) {
    V v = { 3u, 6u };
    V a = v / 3;
    V b = 12 / v;
    VI c = { 0, 0 };
    c = c - 5;
    if (a[0] != 1u || a[1] != 2u) abort();
    if (b[0] != 4u || b[1] != 2u) abort();
    if (c[0] != -5 || c[1] != -5) abort();
    return 0;
}
