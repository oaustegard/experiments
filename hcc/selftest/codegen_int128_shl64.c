// EXPECT-EXIT: 0
// Runtime << with a count >= 64 on a 16-byte operand yields 0 in the low word
// (x86 would otherwise wrap the count modulo 64 and give 1).
int main(void) {
    int n = 64;
    __int128 one = 1;
    if ((one << n) != 0) return 1;
    if ((one << 127) != 0) return 2;
    if ((one << 3) != 8) return 3;
    return 0;
}
