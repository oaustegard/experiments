// EXPECT-EXIT: 0
// A wide (__int128) bitfield yields its full 128-bit value (%rdx:%rax) when read,
// so arithmetic on it (here a division) sees the high word too.
struct s { __int128 f : 119; };
int main(void) {
    struct s v = { ((__int128)-18014398509481984) << 64 };
    __int128 q = v.f / 2;
    if ((long long)(q >> 64) != -9007199254740992LL) return 1;
    if ((unsigned long long)q != 0) return 2;
    return 0;
}
