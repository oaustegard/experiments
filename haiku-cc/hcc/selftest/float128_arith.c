// EXPECT-EXIT: 0
typedef _Float128 F;
F add3(int a, F x, double d, F y, long k, F z, F w, F v, F u, F t) {
    return x + y + z + w + v + u + t + d + a + k;
}
F id(F x) { return x; }
F neg(F x) { return -x; }
int cmp(F a, F b) { return a < b; }
int main(void) {
    F s = add3(1, 2.0Q, 3.0, 4.0Q, 5L, 6.0Q, 7.0Q, 8.0Q, 9.0Q, 10.0Q);
    if (s != 55.0Q) return 1;
    if (id(2.5Q) != 2.5Q) return 2;
    if (neg(id(2.5Q)) != -2.5Q) return 3;
    if (!cmp(1.0Q, 2.0Q) || cmp(2.0Q, 1.0Q)) return 4;
    F acc = 0;
    for (int i = 1; i <= 10; i++) acc += i;
    if (acc != 55) return 5;
    F q = 1.0Q / 3.0Q;
    if ((double)q != 1.0 / 3.0) return 6;
    if ((float)(F)0.5f != 0.5f) return 7;
    F big = (F)1 / (F)1000000000000000000000000000000000000.0Q;
    if (!(big > 0)) return 8;
    if (!(big < 1)) return 9;
    if (!(big != 0)) return 10;
    if (0.0Q == -0.0Q && !(!(F)(-0.0Q))) return 11;
    return 0;
}
