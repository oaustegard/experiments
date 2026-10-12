// EXPECT-EXIT: 0
int main(void) {
    if (__builtin_clrsb(0) != 31 || __builtin_clrsb(-1) != 31) return 1;
    if (__builtin_clrsb(1) != 30 || __builtin_clrsb(-2) != 30) return 2;
    if (__builtin_clrsb(0x40000000) != 0) return 3;
    if (__builtin_clrsb(0x00010000) != 14) return 4;
    if (__builtin_clrsbl(0L) != 63 || __builtin_clrsbl(-1L) != 63) return 5;
    if (__builtin_clrsbll(1LL) != 62 || __builtin_clrsbll(-(1LL << 40)) != 23) return 6;
    return 0;
}
