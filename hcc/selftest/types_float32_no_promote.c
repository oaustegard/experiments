// EXPECT-EXIT: 0
// _Float32 is not subject to default argument promotions: as a variadic
// argument it is passed as float (va_arg reads 4 bytes), and as a K&R
// parameter it stays float.
typedef _Float32 T;
extern void exit(int);
T vafn(T a, ...) {
    __builtin_va_list ap;
    T r;
    __builtin_va_start(ap, a);
    r = a + __builtin_va_arg(ap, T);
    __builtin_va_end(ap);
    return r;
}
T krfn(x) T x; { return x + 1; }
int main(void) {
    volatile T one = 1.0f, neg = -2.5f;
    volatile T r = vafn(one, neg);
    if (r != -1.5f) exit(1);
    r = krfn(neg);
    if (r != -1.5f) exit(2);
    exit(0);
    return 0;
}
