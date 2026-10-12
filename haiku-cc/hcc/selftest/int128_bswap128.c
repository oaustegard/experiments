// EXPECT-EXIT: 0
// __builtin_bswap128 reverses all 16 bytes: each 64-bit half is byte-swapped
// and the halves are exchanged.
int main(void) {
    unsigned __int128 a = __builtin_bswap128((unsigned __int128)2);
    if (a != ((unsigned __int128)2) << 120) return 1;
    unsigned __int128 b = __builtin_bswap128(((unsigned __int128)0xdeadbeefULL) << 64 | 0xcafed00dfeedbac1ULL);
    if (b != ((unsigned __int128)0xc1baedfe0dd0fecaULL << 64 | 0xefbeaddeULL << 32)) return 2;
    return 0;
}
