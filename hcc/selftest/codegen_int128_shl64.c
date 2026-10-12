// EXPECT-EXIT: 0
// Runtime << with a count >= 64 on a 16-byte operand moves bits into the high
// word (x86 would otherwise wrap the count modulo 64 and give 1 << (n & 63)).
int main(void) {
    int n = 64;
    __int128 one = 1;
    if ((one << n) != ((__int128)1 << 64)) return 1;
    if ((((unsigned __int128)one << 127) >> 127) != 1) return 2;
    if ((one << 127) >= 0) return 4;
    if ((one << 3) != 8) return 3;
    if ((one << 65) != ((__int128)1 << 65)) return 5;
    return 0;
}
