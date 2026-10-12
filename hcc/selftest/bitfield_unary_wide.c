// EXPECT-EXIT: 0
// ~ and - on a wide bit-field keep the bit-field's 33-bit precision.
struct X { unsigned long i : 33; };
unsigned long f(struct X *p) { return ~p->i; }
long g(struct X *p) { return -p->i; }
int main(void) {
    struct X x;
    x.i = -1;
    if (f(&x) != 0) return 1;
    if (g(&x) != 1) return 2;
    return 0;
}
