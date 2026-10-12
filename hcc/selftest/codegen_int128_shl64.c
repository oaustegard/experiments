// EXPECT-EXIT: 0
// Runtime << with a count >= 64 on a 16-byte operand yields 0 in the low word
// (x86 would otherwise wrap the count modulo 64 and give 1).
int main(void) {
    int n = 64;
    __int128 one = 1;
    if ((unsigned long)(one << n) != 0) return 1;      // low word
    if ((unsigned long)(one << 127) != 0) return 2;
    if ((one << 3) != 8) return 3;
    if ((one << n) >> 64 != 1) return 4;               // 1 << 64 lives in the high word
    return 0;
}
