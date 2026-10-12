// EXPECT-EXIT: 0
// _Float128 static initializers, literal precision (113-bit significand) and long double round trips.
typedef _Float128 F;
F one = 1.0Q;
F eps = 0x1p-112Q;
F tiny = 0x1p-16382Q;
static F three = 3;
long double big = 0x1.8p+16000L;

int main(void) {
    F x = one + eps;
    if (x == one) return 1;          /* 2^-112 is representable in binary128 */
    if (x - one != eps) return 2;
    if (three != 3) return 3;
    if (tiny == 0) return 4;
    if ((long double)tiny != 0x1p-16382L) return 5;
    if ((F)big != (F)0x1.8p+16000L) return 6;
    if (-tiny >= 0) return 7;
    return 0;
}
