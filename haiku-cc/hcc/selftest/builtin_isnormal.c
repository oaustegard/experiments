// EXPECT-EXIT: 0
// __builtin_isnormal: float/double/_Float16 classification; the operand is
// evaluated exactly once. _Float16 subnormals (below 2^-14) are not normal.
int main(void) {
    int n = 0;
    float f = 1.0f;
    double d = 0.0;
    _Float16 h = (_Float16)0x1p-24, one = (_Float16)1.0;
    if (__builtin_isnormal(f) != 1) return 1;
    if (__builtin_isnormal(d) != 0) return 2;
    if (__builtin_isnormal(h) != 0) return 3;
    if (__builtin_isnormal(one) != 1) return 4;
    if (__builtin_isnormal(__builtin_inf()) != 0) return 5;
    if (__builtin_isnormal(n++ + 1.0) != 1 || n != 1) return 6;
    return 0;
}
