// EXPECT-EXIT: 0
int main(void) {
    unsigned int a = 0x11223344u;
    unsigned long long b = 0x1122334455667788ULL;
    unsigned short c = 0xAABB;
    if (__builtin_bswap32(a) != 0x44332211u) return 1;
    if (__builtin_bswap16(c) != 0xBBAA) return 2;
    if (__builtin_bswap64(b) != 0x8877665544332211ULL) return 3;
    return 0;
}
