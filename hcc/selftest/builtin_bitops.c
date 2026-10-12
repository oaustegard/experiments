// EXPECT-EXIT: 0
int main(void) {
    unsigned int u = 0x00f00000u;
    unsigned long ul = 0x0100000000000000UL;
    if (__builtin_ctz(u) != 20) return 1;
    if (__builtin_clz(u) != 8) return 2;
    if (__builtin_clz(0u) != 32 || __builtin_ctz(0u) != 32) return 3;
    if (__builtin_clzl(ul) != 7 || __builtin_ctzll(ul) != 56) return 4;
    if (__builtin_clzl(0UL) != 64) return 5;
    if (__builtin_popcount(0xF0F0u) != 8 || __builtin_popcountll(~0ULL) != 64) return 6;
    if (__builtin_parity(7u) != 1 || __builtin_parity(3u) != 0) return 7;
    if (__builtin_ffs(0) != 0 || __builtin_ffs(8) != 4 || __builtin_ffsl(1L) != 1) return 8;
    return 0;
}
