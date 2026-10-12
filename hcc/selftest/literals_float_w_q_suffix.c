// EXPECT-EXIT: 0
// GCC's __float80 (w/W) and __float128 (q/Q) floating constant suffixes.
static __float80 a = 2.5w;
static __float128 b = 0.5Q;
int main(void) {
    __float80 x = 2.5W;
    __float128 y = 0.5q;
    if (sizeof(1.0w) != sizeof(long double)) return 3;
    if (x != 2.5L || a != 2.5L) return 1;
    if ((double) y != 0.5 || (double) b != 0.5) return 2;
    return 0;
}
