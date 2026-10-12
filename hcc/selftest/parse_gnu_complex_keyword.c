// EXPECT-EXIT: 0
__complex__ float cf(__complex__ float x) { return x; }
typedef __complex__ double cd;
int main(void) {
    cd d = 2.0;
    __complex__ int i = 3;
    if (cf(1.5f) != 1.5f) return 1;
    if (d != 2.0) return 2;
    return 0;
}
