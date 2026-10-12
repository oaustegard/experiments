// EXPECT-EXIT: 0
// __float80 is the x87 extended type: long double on x86-64 (16 bytes).
static volatile __float80 fv1, fv2;
int main(void) {
    __float80 x = 1.5;
    if (sizeof(__float80) != 16) return 1;
    fv1 = x;
    fv2 = fv1 * 2;
    return (int)fv2 == 3 ? 0 : 2;
}
