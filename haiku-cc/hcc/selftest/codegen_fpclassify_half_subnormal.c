// EXPECT-EXIT: 0
// _Float16 values below 2^-14 are subnormal unless zero. 2^-20 has a zero
// fraction as a float, so it must not be mistaken for zero.
volatile _Float16 sub16 = 0x1p-20f16;
int main(void) {
    if (__builtin_fpclassify(0, 1, 2, 3, 4, sub16) != 3) return 1;
    if (__builtin_fpclassify(0, 1, 2, 3, 4, (_Float16)0.0) != 4) return 2;
    if (__builtin_fpclassify(0, 1, 2, 3, 4, (_Float16)1.0) != 2) return 3;
    return 0;
}
